import os
from smolagents import ToolCallingAgent, OpenAIModel, tool

@tool
def get_exchange_rate(base: str, quote: str) -> str:
    """
    Return a mock FX rate between two currencies.

    Args:
        base: Base currency code like USD
        quote: Quote currency code like CAD
    """
    rates = {
        ("USD", "CAD"): 1.37,
        ("CAD", "USD"): 0.73,
        ("EUR", "USD"): 1.08,
    }
    rate = rates.get((base.upper(), quote.upper()))
    if rate is None:
        return f"No rate found for {base}/{quote}"
    return f"1 {base.upper()} = {rate} {quote.upper()}"

model = OpenAIModel(
    model_id="gemini-3-flash-preview",
    api_key=os.environ["GEMINI_API_KEY"],
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0.2,
)

agent = ToolCallingAgent(
    tools=[get_exchange_rate],
    model=model,
    max_steps=5,
)

result = agent.run("What is the USD to CAD exchange rate? Use the tool.")
print(result)
