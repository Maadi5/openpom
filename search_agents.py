import openai
import asyncio
import aiohttp
import config

# Set your OpenAI API key
openai.api_key = config.openai_key #'YOUR_API_KEY'

# Define a list of prompts for different forums or topics
prompts = [
    "What challenges have travelers faced when carrying unusual electronic devices on flights?",
    "Can you provide examples of passengers who encountered issues with transporting fragrance-related products?",
    "What are common obstacles faced by passengers traveling with innovative tech gadgets?",
    "Share experiences of travelers who had to explain novel devices to airline staff."
]

# Asynchronous function to query the GPT-4 API
async def query_gpt4(prompt):
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {openai.api_key}',
                'Content-Type': 'application/json'
            },
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500
            }
        )
        return await response.json()

# Gather responses from multiple prompts asynchronously
async def gather_responses():
    tasks = [query_gpt4(prompt) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    return responses

# Compile the report from responses
def compile_report(responses):
    report = {}
    for i, response in enumerate(responses):
        content = response['choices'][0]['message']['content']
        report[f"Prompt {i+1}"] = content
    return report

# Main execution using asyncio.run()
if __name__ == "__main__":
    responses = asyncio.run(gather_responses())
    report = compile_report(responses)

    # Print the compiled report
    for prompt, content in report.items():
        print(f"{prompt}:\n{content}\n")