from openai import OpenAI
import json


def read_and_split_file(
    file_path: str, max_chars: int = 4095, max_prompts_kill_switch_num: int = 1
) -> list:
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    chunks = []
    curr_chunk = []
    curr_length = 0

    # Go through the file line-by-line and merge lines
    # into string chunks shorter than max_chars (max prompt length),
    # so they can be sent as separate prompts
    for line in content.split("\n"):
        line_length = len(line)
        if line_length + curr_length > max_chars:
            chunks.append(" ".join(curr_chunk))
            curr_chunk = [line]
            curr_length = line_length
        else:
            curr_chunk.append(line)
            curr_length += line_length

    if curr_chunk:
        chunks.append("\n".join(curr_chunk))

    chunks_num = len(chunks)
    if chunks_num > max_prompts_kill_switch_num:
        raise Exception(
            "SCRIPT INTERRUPTED: the uploaded text file is too large. "
            f"You will be sending more than {max_prompts_kill_switch_num} messages ({chunks_num}) to ChatGPT. "
            "Override this number manually if you are fine with extra tokens ($$$)."
        )

    return chunks


def prepare_messages(text_chunks: list) -> list:
    messages = [
        {
            "role": "system",
            "content": "You are an expert book reader. You will receive a text of a book."
            "Please respond to the questions about the book at the end.",
        }
    ]

    for chunk in text_chunks:
        messages.append({"role": "user", "content": chunk})

    return messages


def send_api_request(messages: list) -> None:
    api_key = json.load(open("config.json"))["API_KEY"]
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Character limit is at 4096, longer text needs to be batched
        messages=messages,
    )

    print(f"Assistant: {completion.choices[0].message.content}")


if __name__ == '__main__':
    text_chunks = read_and_split_file("sample.txt")
    user_input_question = input("Please ask your question about the book: ")
    text_chunks.append(user_input_question)
    messages = prepare_messages(text_chunks)

    send_api_request(messages)
