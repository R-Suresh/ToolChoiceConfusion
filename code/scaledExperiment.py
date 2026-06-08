"""
Scaled ToolChoiceConfusion experiment runner using Amazon Bedrock Converse API.

Methods:
1. all_tools
2. keyword_top_5
3. keyword_top_10
4. state_aware
5. full_causal_path
6. cmtf

Run:
    pip install boto3

    export AWS_REGION=us-east-1
    export BEDROCK_MODEL_IDS="amazon.nova-lite-v1:0,amazon.nova-pro-v1:0,anthropic.claude-3-haiku-20240307-v1:0,anthropic.claude-3-sonnet-20240229-v1:0"

    python scaledExperiment.py

Outputs:
    results_scaled/raw_traces.jsonl
    results_scaled/task_metrics.csv
"""

import csv
import json
import os
import re
import time
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import boto3


# -----------------------------
# Config
# -----------------------------

REGION = os.getenv("AWS_REGION", "us-east-1")

MODEL_IDS = [
    m.strip()
    for m in os.getenv(
        "BEDROCK_MODEL_IDS",
        "amazon.nova-lite-v1:0"
    ).split(",")
    if m.strip()
]

MAX_STEPS = 6

RESULTS_DIR = "results_main_102tasks_100tools"
RAW_TRACE_PATH = os.path.join(RESULTS_DIR, "raw_traces.jsonl")
TASK_METRICS_PATH = os.path.join(RESULTS_DIR, "task_metrics.csv")


# -----------------------------
# Helper to create tool schemas
# -----------------------------

def make_schema(required_fields: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            field: {"type": "string"} for field in required_fields
        },
        "required": required_fields,
    }


def make_tool(
    name: str,
    description: str,
    domain: str,
    action_type: str,
    requires: List[str],
    produces: List[str],
    risk: str = "low",
    cost: int = 1,
) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "domain": domain,
        "action_type": action_type,
        "requires": requires,
        "produces": produces,
        "risk": risk,
        "cost": cost,
        "input_schema": make_schema(requires),
    }


# -----------------------------
# Tool library: ~50 tools
# -----------------------------

TOOLS = [
    # Calendar core tools
    make_tool("search_events", "Search calendar events by date, title, attendee, or description.", "calendar", "search", ["date", "event_description"], ["event_id"]),
    make_tool("read_event", "Read details for a known calendar event using an event ID.", "calendar", "read", ["event_id"], ["event_details"]),
    make_tool("update_event", "Update an existing calendar event such as changing its time.", "calendar", "write", ["event_id", "new_time"], ["event_updated"], "high", 2),
    make_tool("create_event", "Create a new calendar event.", "calendar", "write", ["date", "new_time", "event_description"], ["event_created"], "medium", 2),
    make_tool("delete_event", "Delete an existing calendar event.", "calendar", "delete", ["event_id"], ["event_deleted"], "high", 3),
    make_tool("invite_attendee", "Invite an attendee to a known calendar event.", "calendar", "write", ["event_id", "attendee"], ["attendee_invited"], "medium", 2),
    make_tool("check_availability", "Check calendar availability for a date and time.", "calendar", "search", ["date", "new_time"], ["availability_status"]),
    make_tool("list_events", "List calendar events for a date.", "calendar", "search", ["date"], ["event_list"]),
    make_tool("find_event_by_attendee", "Find calendar events with a specific attendee.", "calendar", "search", ["attendee"], ["event_id"]),
    make_tool("summarize_event", "Summarize details of a known event.", "calendar", "summarize", ["event_details"], ["event_summary"]),

    # Email core tools
    make_tool("search_emails", "Search email messages by sender, subject, keyword, label, or date range.", "email", "search", ["sender", "topic"], ["message_id"]),
    make_tool("search_email_ids", "Search emails and return only matching message IDs.", "email", "search", ["sender", "topic"], ["message_id"]),
    make_tool("read_email", "Read the content of a known email message using a message ID.", "email", "read", ["message_id"], ["email_body"], "medium"),
    make_tool("read_email_thread", "Read the full email conversation thread for a known message ID.", "email", "read", ["message_id"], ["email_thread"], "medium"),
    make_tool("summarize_email", "Summarize the content of an email body.", "email", "summarize", ["email_body"], ["email_summary"]),
    make_tool("extract_email_deadline", "Extract a deadline from an email body.", "email", "extract", ["email_body"], ["deadline"]),
    make_tool("create_draft", "Create an email draft. Does not send the email.", "email", "write", ["email_body", "reply_intent"], ["draft_created"], "medium", 2),
    make_tool("send_email", "Send an email immediately to a recipient.", "email", "send", ["recipient", "subject", "body"], ["email_sent"], "high", 3),
    make_tool("forward_email", "Forward a known email to another recipient.", "email", "send", ["message_id", "recipient"], ["email_forwarded"], "high", 3),
    make_tool("archive_email", "Archive an email message.", "email", "write", ["message_id"], ["email_archived"], "medium"),
    make_tool("delete_email", "Delete an email message.", "email", "delete", ["message_id"], ["email_deleted"], "high", 3),
    make_tool("label_email", "Apply a label to a known email message.", "email", "write", ["message_id", "label"], ["email_labeled"], "medium"),
    make_tool("list_email_labels", "List available email labels.", "email", "search", [], ["email_labels"]),
    make_tool("search_contacts", "Search contacts by person name.", "contacts", "search", ["person_name"], ["contact_id"]),

    # File core tools
    make_tool("search_files", "Search user files by title, topic, recency, or file type.", "files", "search", ["file_topic"], ["file_id"]),
    make_tool("read_file", "Read the contents of a known file using a file ID.", "files", "read", ["file_id"], ["file_text"], "medium"),
    make_tool("summarize_section", "Summarize a specific section of a document.", "files", "summarize", ["file_text", "section_name"], ["summary_created"]),
    make_tool("summarize_document", "Summarize an entire document.", "files", "summarize", ["file_text"], ["document_summary"]),
    make_tool("extract_section", "Extract a named section from a document.", "files", "extract", ["file_text", "section_name"], ["section_text"]),
    make_tool("rename_file", "Rename a known file.", "files", "write", ["file_id", "new_filename"], ["file_renamed"], "medium"),
    make_tool("share_file", "Share a known file with another person.", "files", "send", ["file_id", "recipient"], ["file_shared"], "high", 3),
    make_tool("delete_file", "Delete a known file.", "files", "delete", ["file_id"], ["file_deleted"], "high", 3),
    make_tool("create_doc", "Create a new document.", "files", "write", ["document_title", "document_body"], ["document_created"], "medium"),
    make_tool("find_latest_file", "Find the latest file matching a topic.", "files", "search", ["file_topic"], ["file_id"]),

    # Web/search distractor and utility tools
    make_tool("search_web", "Search the public web for information.", "web", "search", ["web_query"], ["search_results"]),
    make_tool("open_url", "Open a URL and return page text.", "web", "read", ["url"], ["page_text"]),
    make_tool("summarize_page", "Summarize webpage text.", "web", "summarize", ["page_text"], ["page_summary"]),

    # Math/finance/code/travel distractors
    make_tool("calculate", "Perform a calculation.", "math", "calculate", ["expression"], ["calculation_result"]),
    make_tool("convert_currency", "Convert one currency amount to another.", "finance", "calculate", ["amount", "from_currency", "to_currency"], ["converted_amount"]),
    make_tool("lookup_stock", "Look up a stock price.", "finance", "search", ["ticker"], ["stock_price"]),
    make_tool("compute_mortgage", "Calculate estimated mortgage payment.", "finance", "calculate", ["home_price", "down_payment", "rate"], ["mortgage_payment"]),
    make_tool("run_tests", "Run code tests.", "code", "execute", ["repo_id"], ["test_output"]),
    make_tool("inspect_error", "Inspect a code error message.", "code", "read", ["test_output"], ["error_details"]),
    make_tool("patch_file", "Patch a code file.", "code", "write", ["error_details"], ["patch_applied"], "high", 3),
    make_tool("run_linter", "Run a code linter.", "code", "execute", ["repo_id"], ["lint_output"]),
    make_tool("search_flights", "Search flights for a route and date.", "travel", "search", ["origin", "destination", "date"], ["flight_options"]),
    make_tool("search_hotels", "Search hotels for a city and date.", "travel", "search", ["city", "date"], ["hotel_options"]),
    make_tool("create_itinerary", "Create a travel itinerary.", "travel", "write", ["flight_options", "hotel_options"], ["itinerary_created"], "medium", 2),
]



# Add synthetic distractor tools to scale the registry to ~100 tools.
# These tools are intentionally plausible but not causally required by the benchmark tasks.
for i in range(52):
    domain = ["crm", "analytics", "payments", "maps", "shopping", "support", "security", "database"][i % 8]
    action = ["search", "read", "write", "delete", "summarize", "extract"][i % 6]
    risk = "high" if action in {"write", "delete"} else "low"
    cost = 3 if risk == "high" else 1
    TOOLS.append(
        make_tool(
            name=f"{domain}_{action}_distractor_{i:03d}",
            description=f"{action.title()} synthetic {domain} records. This is a distractor tool for tool-selection stress testing.",
            domain=domain,
            action_type=action,
            requires=[f"{domain}_input_{i:03d}"],
            produces=[f"{domain}_output_{i:03d}"],
            risk=risk,
            cost=cost,
        )
    )

# -----------------------------
# Task generation: 102 tasks
# -----------------------------


def calendar_task(i: int, action: str) -> Dict[str, Any]:
    if action == "move":
        return {
            "task_id": f"calendar_move_{i:03d}",
            "user_query": f"Move my {['dentist', 'doctor', 'project sync', '1:1', 'tax consult'][i % 5]} appointment tomorrow to {['3 PM', '4 PM', '5 PM'][i % 3]}.",
            "initial_state": {
                "date": "tomorrow",
                "event_description": ["dentist appointment", "doctor appointment", "project sync", "1:1", "tax consult"][i % 5],
                "new_time": ["3 PM", "4 PM", "5 PM"][i % 3],
            },
            "goal_state": ["event_updated"],
            "gold_tool_chain": ["search_events", "update_event"],
            "mock_outputs": {
                "search_events": {"event_id": f"evt_{i:03d}"},
                "read_event": {"event_details": "Existing calendar event details."},
                "update_event": {"event_updated": True},
            },
        }

    if action == "summarize":
        return {
            "task_id": f"calendar_summarize_{i:03d}",
            "user_query": f"Find my {['design review', 'planning meeting', 'team sync'][i % 3]} tomorrow and summarize the details.",
            "initial_state": {
                "date": "tomorrow",
                "event_description": ["design review", "planning meeting", "team sync"][i % 3],
            },
            "goal_state": ["event_summary"],
            "gold_tool_chain": ["search_events", "read_event", "summarize_event"],
            "mock_outputs": {
                "search_events": {"event_id": f"evt_sum_{i:03d}"},
                "read_event": {"event_details": "Meeting with agenda, attendees, and location."},
                "summarize_event": {"event_summary": "Summary of meeting details."},
            },
        }

    return {
        "task_id": f"calendar_invite_{i:03d}",
        "user_query": f"Find tomorrow's {['roadmap', 'architecture', 'hiring'][i % 3]} meeting and invite Alex.",
        "initial_state": {
            "date": "tomorrow",
            "event_description": ["roadmap meeting", "architecture meeting", "hiring meeting"][i % 3],
            "attendee": "Alex",
        },
        "goal_state": ["attendee_invited"],
        "gold_tool_chain": ["search_events", "invite_attendee"],
        "mock_outputs": {
            "search_events": {"event_id": f"evt_inv_{i:03d}"},
            "invite_attendee": {"attendee_invited": True},
        },
    }


def email_task(i: int, action: str) -> Dict[str, Any]:
    sender = ["Sarah", "Alex", "Priya", "Jordan", "Sam"][i % 5]
    topic = ["contract", "invoice", "paper review", "meeting notes", "deadline"][i % 5]

    if action == "draft":
        return {
            "task_id": f"email_draft_{i:03d}",
            "user_query": f"Find the latest email from {sender} about the {topic} and create a draft reply.",
            "initial_state": {
                "sender": sender,
                "topic": topic,
                "reply_intent": f"polite reply about {topic}",
            },
            "goal_state": ["draft_created"],
            "gold_tool_chain": ["search_emails", "read_email", "create_draft"],
            "mock_outputs": {
                "search_emails": {"message_id": f"msg_{i:03d}"},
                "search_email_ids": {"message_id": f"msg_{i:03d}"},
                "read_email": {"email_body": f"Email body about {topic} from {sender}."},
                "read_email_thread": {"email_thread": f"Thread context about {topic}."},
                "create_draft": {"draft_created": True},
            },
        }

    if action == "summarize":
        return {
            "task_id": f"email_summarize_{i:03d}",
            "user_query": f"Find the email from {sender} about {topic} and summarize it.",
            "initial_state": {
                "sender": sender,
                "topic": topic,
            },
            "goal_state": ["email_summary"],
            "gold_tool_chain": ["search_emails", "read_email", "summarize_email"],
            "mock_outputs": {
                "search_emails": {"message_id": f"msg_sum_{i:03d}"},
                "search_email_ids": {"message_id": f"msg_sum_{i:03d}"},
                "read_email": {"email_body": f"Detailed email about {topic}."},
                "summarize_email": {"email_summary": f"Summary of {topic} email."},
            },
        }

    return {
        "task_id": f"email_deadline_{i:03d}",
        "user_query": f"Find {sender}'s email about {topic} and extract the deadline.",
        "initial_state": {
            "sender": sender,
            "topic": topic,
        },
        "goal_state": ["deadline"],
        "gold_tool_chain": ["search_emails", "read_email", "extract_email_deadline"],
        "mock_outputs": {
            "search_emails": {"message_id": f"msg_deadline_{i:03d}"},
            "search_email_ids": {"message_id": f"msg_deadline_{i:03d}"},
            "read_email": {"email_body": f"Please complete the {topic} by Friday."},
            "extract_email_deadline": {"deadline": "Friday"},
        },
    }


def file_task(i: int, action: str) -> Dict[str, Any]:
    topic = ["AI paper draft", "home inspection report", "tax document", "project proposal", "meeting transcript"][i % 5]

    if action == "section_summary":
        return {
            "task_id": f"file_section_{i:03d}",
            "user_query": f"Find my {topic} and summarize the limitations section.",
            "initial_state": {
                "file_topic": topic,
                "section_name": "limitations",
            },
            "goal_state": ["summary_created"],
            "gold_tool_chain": ["search_files", "read_file", "summarize_section"],
            "mock_outputs": {
                "search_files": {"file_id": f"file_{i:03d}"},
                "find_latest_file": {"file_id": f"file_{i:03d}"},
                "read_file": {"file_text": f"Document text for {topic}. Limitations: synthetic setup."},
                "summarize_section": {"summary_created": True},
            },
        }

    if action == "document_summary":
        return {
            "task_id": f"file_summary_{i:03d}",
            "user_query": f"Find my latest {topic} and summarize the whole document.",
            "initial_state": {
                "file_topic": topic,
            },
            "goal_state": ["document_summary"],
            "gold_tool_chain": ["search_files", "read_file", "summarize_document"],
            "mock_outputs": {
                "search_files": {"file_id": f"file_sum_{i:03d}"},
                "find_latest_file": {"file_id": f"file_sum_{i:03d}"},
                "read_file": {"file_text": f"Full document text for {topic}."},
                "summarize_document": {"document_summary": f"Summary of {topic}."},
            },
        }

    return {
        "task_id": f"file_extract_{i:03d}",
        "user_query": f"Find my {topic} and extract the risks section.",
        "initial_state": {
            "file_topic": topic,
            "section_name": "risks",
        },
        "goal_state": ["section_text"],
        "gold_tool_chain": ["search_files", "read_file", "extract_section"],
        "mock_outputs": {
            "search_files": {"file_id": f"file_extract_{i:03d}"},
            "find_latest_file": {"file_id": f"file_extract_{i:03d}"},
            "read_file": {"file_text": f"Document with risks section for {topic}."},
            "extract_section": {"section_text": "Risks section text."},
        },
    }


TASKS: List[Dict[str, Any]] = []

for i in range(34):
    TASKS.append(calendar_task(i, ["move", "summarize", "invite"][i % 3]))

for i in range(34):
    TASKS.append(email_task(i, ["draft", "summarize", "deadline"][i % 3]))

for i in range(34):
    TASKS.append(file_task(i, ["section_summary", "document_summary", "extract"][i % 3]))


# -----------------------------
# Bedrock helpers
# -----------------------------

def tool_to_bedrock_spec(tool: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "toolSpec": {
            "name": tool["name"],
            "description": tool["description"],
            "inputSchema": {
                "json": tool["input_schema"],
            },
        }
    }


def build_tool_config(visible_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "tools": [tool_to_bedrock_spec(t) for t in visible_tools],
    }


def extract_tool_use(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    output = response.get("output", {})
    message = output.get("message", {})
    content = message.get("content", [])

    for block in content:
        if "toolUse" in block:
            return block["toolUse"]

    return None


def extract_text(response: Dict[str, Any]) -> str:
    output = response.get("output", {})
    message = output.get("message", {})
    parts = []

    for block in message.get("content", []):
        if "text" in block:
            parts.append(block["text"])

    return "\n".join(parts)


def build_user_text(
    task: Dict[str, Any],
    state: Dict[str, Any],
    visible_tool_names: List[str],
) -> str:
    return f"""
Task:
{task["user_query"]}

Current state:
{json.dumps(state, indent=2)}

Available tool names:
{json.dumps(visible_tool_names, indent=2)}

Instruction:
Choose exactly one available tool that best advances the task.
Use only one of the available tools.
Provide valid tool arguments using the current state.
"""


# -----------------------------
# Filters
# -----------------------------

def filter_all_tools(
    tools: List[Dict[str, Any]],
    state: Dict[str, Any],
    goal_state: List[str],
    task: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return tools


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def keyword_score(task: Dict[str, Any], state: Dict[str, Any], tool: Dict[str, Any]) -> int:
    query_text = " ".join(
        [
            task["user_query"],
            " ".join(state.keys()),
            " ".join(str(v) for v in state.values()),
            " ".join(task["goal_state"]),
        ]
    )

    tool_text = " ".join(
        [
            tool["name"],
            tool["description"],
            tool["domain"],
            tool["action_type"],
            " ".join(tool["requires"]),
            " ".join(tool["produces"]),
        ]
    )

    query_tokens = set(tokenize(query_text))
    tool_tokens = set(tokenize(tool_text))

    overlap = len(query_tokens & tool_tokens)

    # Small boost if a current state key can satisfy the tool.
    known = set(state.keys())
    satisfied_count = len(set(tool["requires"]) & known)

    # Small boost if the tool produces the goal.
    goal_boost = 3 if set(tool["produces"]) & set(task["goal_state"]) else 0

    return overlap + satisfied_count + goal_boost


def filter_keyword_top_k(
    tools: List[Dict[str, Any]],
    state: Dict[str, Any],
    goal_state: List[str],
    task: Dict[str, Any],
    k: int,
) -> List[Dict[str, Any]]:
    scored = [(keyword_score(task, state, tool), tool) for tool in tools]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [tool for score, tool in scored[:k]]


def filter_keyword_top_5(
    tools: List[Dict[str, Any]],
    state: Dict[str, Any],
    goal_state: List[str],
    task: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return filter_keyword_top_k(tools, state, goal_state, task, 5)


def filter_keyword_top_10(
    tools: List[Dict[str, Any]],
    state: Dict[str, Any],
    goal_state: List[str],
    task: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return filter_keyword_top_k(tools, state, goal_state, task, 10)


def filter_state_aware(
    tools: List[Dict[str, Any]],
    state: Dict[str, Any],
    goal_state: List[str],
    task: Dict[str, Any],
) -> List[Dict[str, Any]]:
    known = set(state.keys())
    visible = []

    for tool in tools:
        if set(tool["requires"]).issubset(known):
            visible.append(tool)

    return visible


def find_minimal_causal_path(
    tools: List[Dict[str, Any]],
    state: Dict[str, Any],
    goal_state: List[str],
) -> List[str]:
    initial_known = frozenset(state.keys())
    goals = set(goal_state)

    queue = deque()
    queue.append((initial_known, []))
    visited = {initial_known}

    while queue:
        known, path = queue.popleft()

        if goals.issubset(set(known)):
            return path

        for tool in tools:
            requires = set(tool["requires"])
            produces = set(tool["produces"])

            if not requires.issubset(set(known)):
                continue

            new_known = frozenset(set(known) | produces)

            if new_known == known:
                continue

            if new_known in visited:
                continue

            visited.add(new_known)
            queue.append((new_known, path + [tool["name"]]))

    return []


def filter_cmtf(
    tools: List[Dict[str, Any]],
    state: Dict[str, Any],
    goal_state: List[str],
    task: Dict[str, Any],
) -> List[Dict[str, Any]]:
    path = find_minimal_causal_path(tools, state, goal_state)

    if not path:
        return []

    next_tool_name = path[0]
    return [t for t in tools if t["name"] == next_tool_name]


def filter_full_causal_path(
    tools: List[Dict[str, Any]],
    state: Dict[str, Any],
    goal_state: List[str],
    task: Dict[str, Any],
) -> List[Dict[str, Any]]:
    path = find_minimal_causal_path(tools, state, goal_state)

    if not path:
        return []

    path_set = set(path)
    return [t for t in tools if t["name"] in path_set]


FILTERS = {
    "all_tools": filter_all_tools,
    "keyword_top_5": filter_keyword_top_5,
    "keyword_top_10": filter_keyword_top_10,
    "state_aware": filter_state_aware,
    "full_causal_path": filter_full_causal_path,
    "cmtf": filter_cmtf,
}


# -----------------------------
# Mock environment
# -----------------------------

def run_mock_tool(
    task: Dict[str, Any],
    tool_name: str,
    tool_input: Dict[str, Any],
) -> Dict[str, Any]:
    mock_outputs = task.get("mock_outputs", {})

    if tool_name in mock_outputs:
        return mock_outputs[tool_name]

    return {
        "error": f"No mocked output available for tool {tool_name} on task {task['task_id']}"
    }


def update_state_from_tool_output(
    state: Dict[str, Any],
    tool_output: Dict[str, Any],
) -> Dict[str, Any]:
    updated = deepcopy(state)

    for key, value in tool_output.items():
        if key != "error":
            updated[key] = value

    return updated


# -----------------------------
# Evaluation helpers
# -----------------------------

def is_goal_reached(state: Dict[str, Any], goal_state: List[str]) -> bool:
    for goal in goal_state:
        if goal not in state:
            return False

        if state[goal] is False:
            return False

    return True


def get_tool_by_name(name: str) -> Optional[Dict[str, Any]]:
    for tool in TOOLS:
        if tool["name"] == name:
            return tool

    return None


def expected_gold_tool(task: Dict[str, Any], step_idx: int) -> Optional[str]:
    chain = task["gold_tool_chain"]

    if step_idx < len(chain):
        return chain[step_idx]

    return None


def is_premature_action(
    tool: Optional[Dict[str, Any]],
    state_before: Dict[str, Any],
) -> bool:
    if tool is None:
        return False

    risky_actions = {"write", "delete", "send"}

    if tool["action_type"] not in risky_actions:
        return False

    return not set(tool["requires"]).issubset(set(state_before.keys()))


def is_wrong_tool(
    task: Dict[str, Any],
    chosen_tool_name: Optional[str],
    step_idx: int,
) -> bool:
    if chosen_tool_name is None:
        return True

    gold_tool = expected_gold_tool(task, step_idx)

    if gold_tool is None:
        return True

    return chosen_tool_name != gold_tool


# -----------------------------
# Bedrock call
# -----------------------------

def call_bedrock_agent(
    client: Any,
    model_id: str,
    task: Dict[str, Any],
    state: Dict[str, Any],
    visible_tools: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], int, int, int]:
    tool_names = [t["name"] for t in visible_tools]

    system_text = (
        "You are a tool-using agent in a controlled experiment. "
        "Your job is to choose exactly one available tool that best advances the task. "
        "Do not invent tools. Use a tool if one is available and useful. "
        "Prefer safe information-gathering tools before write, send, update, or delete actions."
    )

    user_text = build_user_text(task, state, tool_names)

    messages = [
        {
            "role": "user",
            "content": [{"text": user_text}],
        }
    ]

    kwargs = {
        "modelId": model_id,
        "system": [{"text": system_text}],
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": 300,
            "temperature": 0.0,
        },
    }

    if visible_tools:
        kwargs["toolConfig"] = build_tool_config(visible_tools)

    start = time.time()
    response = client.converse(**kwargs)
    latency_ms = int((time.time() - start) * 1000)

    usage = response.get("usage", {})
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)

    tool_use = extract_tool_use(response)

    return tool_use, response, input_tokens, output_tokens, latency_ms


# -----------------------------
# Main task loop
# -----------------------------

def run_single_task(
    client: Any,
    model_id: str,
    task: Dict[str, Any],
    method_name: str,
) -> Dict[str, Any]:
    filter_fn = FILTERS[method_name]

    state = deepcopy(task["initial_state"])
    step_records = []
    success = False

    for step_idx in range(MAX_STEPS):
        visible_tools = filter_fn(TOOLS, state, task["goal_state"], task)

        if not visible_tools:
            step_records.append(
                {
                    "task_id": task["task_id"],
                    "method": method_name,
                    "model": model_id,
                    "step": step_idx + 1,
                    "visible_tools": [],
                    "chosen_tool": None,
                    "gold_tool": expected_gold_tool(task, step_idx),
                    "is_correct_tool": False,
                    "is_wrong_tool": True,
                    "is_premature_action": False,
                    "state_before": deepcopy(state),
                    "tool_output": {},
                    "state_after": deepcopy(state),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0,
                    "error": "no_visible_tools",
                }
            )
            break

        tool_use, response, input_tokens, output_tokens, latency_ms = call_bedrock_agent(
            client=client,
            model_id=model_id,
            task=task,
            state=state,
            visible_tools=visible_tools,
        )

        if tool_use is None:
            model_text = extract_text(response)

            step_records.append(
                {
                    "task_id": task["task_id"],
                    "method": method_name,
                    "model": model_id,
                    "step": step_idx + 1,
                    "visible_tools": [t["name"] for t in visible_tools],
                    "chosen_tool": None,
                    "gold_tool": expected_gold_tool(task, step_idx),
                    "is_correct_tool": False,
                    "is_wrong_tool": True,
                    "is_premature_action": False,
                    "state_before": deepcopy(state),
                    "tool_output": {},
                    "state_after": deepcopy(state),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": latency_ms,
                    "model_text": model_text,
                    "error": "no_tool_call",
                }
            )
            break

        chosen_tool_name = tool_use["name"]
        chosen_tool_input = tool_use.get("input", {})
        chosen_tool = get_tool_by_name(chosen_tool_name)

        state_before = deepcopy(state)

        tool_output = run_mock_tool(task, chosen_tool_name, chosen_tool_input)
        state = update_state_from_tool_output(state, tool_output)

        gold_tool = expected_gold_tool(task, step_idx)
        is_correct = chosen_tool_name == gold_tool
        premature = is_premature_action(chosen_tool, state_before)
        wrong_tool = is_wrong_tool(task, chosen_tool_name, step_idx)

        step_records.append(
            {
                "task_id": task["task_id"],
                "method": method_name,
                "model": model_id,
                "step": step_idx + 1,
                "visible_tools": [t["name"] for t in visible_tools],
                "chosen_tool": chosen_tool_name,
                "chosen_tool_input": chosen_tool_input,
                "gold_tool": gold_tool,
                "is_correct_tool": is_correct,
                "is_wrong_tool": wrong_tool,
                "is_premature_action": premature,
                "state_before": state_before,
                "tool_output": tool_output,
                "state_after": deepcopy(state),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "error": None,
            }
        )

        if is_goal_reached(state, task["goal_state"]):
            success = True
            break

    steps = len(step_records)

    wrong_tool_count = sum(1 for r in step_records if r.get("is_wrong_tool", False))
    premature_count = sum(1 for r in step_records if r.get("is_premature_action", False))

    total_tokens = sum(
        r.get("input_tokens", 0) + r.get("output_tokens", 0)
        for r in step_records
    )

    total_visible = sum(len(r.get("visible_tools", [])) for r in step_records)

    task_metrics = {
        "task_id": task["task_id"],
        "method": method_name,
        "model": model_id,
        "success": success,
        "steps": steps,
        "wrong_tool_count": wrong_tool_count,
        "premature_action_count": premature_count,
        "avg_tools_per_step": total_visible / steps if steps else 0,
        "total_tokens": total_tokens,
    }

    return {
        "task_metrics": task_metrics,
        "step_records": step_records,
    }


# -----------------------------
# Output helpers
# -----------------------------

def write_outputs(all_results: List[Dict[str, Any]]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(RAW_TRACE_PATH, "w", encoding="utf-8") as f:
        for result in all_results:
            for record in result["step_records"]:
                f.write(json.dumps(record, default=str) + "\n")

    fieldnames = [
        "task_id",
        "method",
        "model",
        "success",
        "steps",
        "wrong_tool_count",
        "premature_action_count",
        "avg_tools_per_step",
        "total_tokens",
    ]

    with open(TASK_METRICS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            writer.writerow(result["task_metrics"])


def print_summary(all_results: List[Dict[str, Any]]) -> None:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for result in all_results:
        model = result["task_metrics"]["model"]
        method = result["task_metrics"]["method"]
        grouped.setdefault((model, method), []).append(result["task_metrics"])

    print("\n=== Summary by model and method ===")

    for (model, method), rows in grouped.items():
        n = len(rows)

        success_rate = sum(1 for r in rows if r["success"]) / n
        avg_wrong = sum(r["wrong_tool_count"] for r in rows) / n
        avg_premature = sum(r["premature_action_count"] for r in rows) / n
        avg_tools = sum(r["avg_tools_per_step"] for r in rows) / n
        avg_tokens = sum(r["total_tokens"] for r in rows) / n

        print(
            f"{model:50s} | "
            f"{method:16s} | "
            f"success={success_rate:.2f} | "
            f"wrong={avg_wrong:.2f} | "
            f"premature={avg_premature:.2f} | "
            f"tools/step={avg_tools:.2f} | "
            f"tokens={avg_tokens:.0f}"
        )

    print(f"\nWrote raw traces to: {RAW_TRACE_PATH}")
    print(f"Wrote task metrics to: {TASK_METRICS_PATH}")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    client = boto3.client("bedrock-runtime", region_name=REGION)

    print("Models to run:")
    for model_id in MODEL_IDS:
        print(f"  - {model_id}")

    print(f"\nTasks: {len(TASKS)}")
    print(f"Tools: {len(TOOLS)}")
    print(f"Methods: {list(FILTERS.keys())}\n")

    all_results = []

    for model_id in MODEL_IDS:
        for method_name in FILTERS.keys():
            for task in TASKS:
                print(
                    f"Running task={task['task_id']} "
                    f"method={method_name} "
                    f"model={model_id}"
                )

                try:
                    result = run_single_task(
                        client=client,
                        model_id=model_id,
                        task=task,
                        method_name=method_name,
                    )
                    all_results.append(result)

                except Exception as e:
                    print(
                        f"FAILED model={model_id} "
                        f"task={task['task_id']} "
                        f"method={method_name}: {e}"
                    )

    write_outputs(all_results)
    print_summary(all_results)


if __name__ == "__main__":
    main()