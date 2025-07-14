from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="YOUR_AZURE_OPENAI_KEY",
    base_url="https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1",
    api_version="preview"
)

# Upload the PDF
with open("yourfile.pdf", "rb") as f:
    file = client.files.create(file=f, purpose="user_data")
file_id = file.id

response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": file_id},
                {"type": "input_text", "text": "Extract the content from the file provided without altering it. Just output its exact content and nothing else."}
            ]
        }
    ]
)
print(response.output_text)
