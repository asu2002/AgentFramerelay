from agentframerelay import Agent, tool

@tool
def search_customer(customer_id: str) -> dict:
    """Retrieve customer information."""
    return {"customer_id": customer_id, "status": "active"}

agent = Agent(
    name="customer-agent",
    instructions="Help users retrieve customer information.",
    strategy="react",
    runtime="mock",
    tools=[search_customer],
)

print(agent.run({"input": "Find customer 123"}))
print(agent.capabilities())
