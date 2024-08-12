import os
import requests

from langchain import hub
from langchain_core.pydantic_v1 import BaseModel, Field, EmailStr
from langchain_core.tools import tool, BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from typing import Type, Optional, Dict
from utils.agent_helpers import _prompt_text_loader

class ProjectBrief(BaseModel):
    job_headline: str = Field(..., description="a comprehensive explainatory headline title for the job", min_length=50, max_length=100)
    job_details: str = Field(..., description="Any specific instructions or special details about the job. This is often, but not always, about the tools needed for the job and who will supply them. For example 'You already have the paint and supplies for each room' or 'the pro will need to bring his own supplies to clean your yard'")
    job_address: str = Field(..., description="What is the full address of where the job is taking place")
    time_estimate: str = Field(..., description="Based on the information, how long do we expect the job to take")
    cost_estimate: str = Field(..., description="Based on the information, how much do we expect the job to cost")

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
book_appointment_prompt
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
@tool("create_brief", args_schema=ProjectBrief)  
def create_brief(job_headline: str, job_details: str, job_address: str, time_estimate: str, cost_estimate: str) -> Dict[str, str]:
    project_brief = {
        "job_headline": job_headline,
        "job_details": job_details,  # Plain string
        "job_address": job_address,
        "time_estimate": time_estimate,  # Plain string
        "cost_estimate": cost_estimate  # Plain string
    }
    return project_brief