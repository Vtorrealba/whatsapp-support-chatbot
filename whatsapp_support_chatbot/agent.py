class Agent:
    def __init__(self):
        self.agent = None
        self.memory = []
        self.context = []
        self.response = ""
        self.user_input = ""
        self.assistant_input = ""

    def get_response(self, user_input):
        self.user_input = user_input
        self.assistant_input = self.agent.run(self.user_input)
        return self.assistant_input