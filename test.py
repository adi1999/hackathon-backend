import os
import re
import subprocess
import sys

import tiktoken

from doc_comments_ai.constants import Language


def get_programming_language(file_extension: str) -> Language:
    language_mapping = {
        ".py": Language.PYTHON,
        ".js": Language.JAVASCRIPT,
        ".jsx": Language.JAVASCRIPT,
        ".mjs": Language.JAVASCRIPT,
        ".cjs": Language.JAVASCRIPT,
        ".ts": Language.TYPESCRIPT,
        ".tsx": Language.TYPESCRIPT,
        ".java": Language.JAVA,
        ".kt": Language.KOTLIN,
        ".rs": Language.RUST,
        ".go": Language.GO,
        ".cpp": Language.CPP,
        ".c": Language.C,
        ".cs": Language.C_SHARP,
    }
    return language_mapping.get(file_extension, Language.UNKNOWN)


def get_file_extension(file_name: str) -> str:
    return os.path.splitext(file_name)[-1]


def write_code_snippet_to_file(
    file_path: str, original_code: str, modified_code: str
) -> None:
    with open(file_path, "r") as file:
        file_content = file.read()
        start_pos = file_content.find(original_code)
        if start_pos != -1:
            end_pos = start_pos + len(original_code)
            indentation = file_content[:start_pos].split("\n")[-1]
            modeified_lines = modified_code.split("\n")
            indented_modified_lines = [indentation + line for line in modeified_lines]
            indented_modified_code = "\n".join(indented_modified_lines)
            modified_content = (
                file_content[:start_pos].rstrip()
                + "\n"
                + indented_modified_code
                + file_content[end_pos:]
            )
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(modified_content)


def extract_content_from_markdown_code_block(markdown_code_block) -> str:
    pattern = r"```(?:[a-zA-Z0-9]+)?\n(.*?)```"
    match = re.search(pattern, markdown_code_block, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return markdown_code_block.strip()


def get_bold_text(text):
    return f"\033[01m{text}\033[0m"


def has_unstaged_changes(file):
    try:
        subprocess.check_output(["git", "diff", "--quiet", file])
        return False
    except subprocess.CalledProcessError:
        return True


def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokenized = encoding.encode(text)
    return len(tokenized)


def is_openai_api_key_available():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not found.")


def is_azure_openai_environment_available():
    azure_api_base = os.environ.get("AZURE_API_BASE")
    azure_api_key = os.environ.get("AZURE_API_KEY")
    azure_api_version = os.environ.get("AZURE_API_VERSION")
    if not azure_api_base or not azure_api_key or not azure_api_version:
        if not azure_api_base:
            print("AZURE_API_BASE not found.")
        if not azure_api_key:
            print("AZURE_API_KEY not found.")
        if not azure_api_version:
            print("AZURE_API_VERSION not found.")
        sys.exit("Please set the environment variables for Azure OpenAI deployment.")
