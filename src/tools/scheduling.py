import os
import requests

from langchain import hub
from langchain_core.pydantic_v1 import BaseModel, Field, EmailStr
from langchain_core.tools import tool, BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from typing import Type, Optional, Dict
from utils.agent_helpers import _prompt_text_loader

class CancelBooking(BaseModel):
    booking_id: int = Field(..., description="")

class CheckCalendar(BaseModel): 
    start_date: str = Field(..., description="The start date to check availability in the calendar. input should be a future date in yyyy-mm-dd format")
    end_date: str = Field(..., description="The end date to check availability in the calendar. input should be a future date in yyyy-mm-dd format this should be 2 days after start_date")

class AppointmentBooking(BaseModel):
    name: str = Field(description="The name of the person or company booking the appointment", min_length=2, max_length=100)
    email: EmailStr = Field(..., description="The email of the person or company booking the appointment")
    date: str = Field(..., description="The date and time of the appointment in ISO 8601 format (e.g., 2024-07-27T13:00:00.000Z)")
    address: str = Field(..., description="the place where the service will pe provided")


@tool("check_calendar", args_schema=CheckCalendar)
def check_calendar(start_date: str, end_date: str) -> Dict:
    """
    Check the availability of a date provided by the user in the calendar. 
    The date might not come to you in the yyyy-mm-dd format. in that case,
    convert it to the right format before calling this tool.  
    """
    webhook_url = os.getenv("CC_WEBHOOK_URL")
    response = requests.post(webhook_url, json={"start_date": start_date, "end_date": end_date})
    
    if response.status_code == 200:
        response_data = response.json()
        slots = response_data.get("data", {}).get("slots", {})
        
        if not slots:
            return {"error": "No availability found"}

        # Extracting and flattening the time values
        flat_times = [slot["time"] for day_slots in slots.values() for slot in day_slots]
        
        return {
            "description": response_data.get("description", ""),
            "instruction": "to proceed with a booking, copy one of the below valid timestamps",
            "available_times": flat_times
        }
    else:
        return {"error": f"{response.status_code} bad request"}

book_appointment_prompt = hub.pull("book_appointment_tool")
book_appointment_tool_description = _prompt_text_loader(book_appointment_prompt)      
class BookAppointmentTool(BaseTool):
    name = "book_appointment"
    description = book_appointment_tool_description
    args_schema: Type[BaseModel] = AppointmentBooking

    def _run(
        self, name: str, email: str, date: str, address:str,run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Book an appointment."""
        webhook_url = os.getenv("PCP_WEBHOOK_URL")
        payload = {
            "name": name,
            "email": email,
            "date": date,
            "address": address,
        }
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            return "Appointment successfully booked."
        else:
            return f"Failed to book appointment. Status code: {response.status_code}"

    async def _arun(
        self, name: str, email: str, date: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Asynchronously book an appointment."""
        # Implement async version if needed
        raise NotImplementedError("Async version not implemented")
      
book_appointment = BookAppointmentTool()


# @tool("cancel_booking", args_schema=CancelBooking)
# def cancel_booking(booking_id: int) -> str
    
