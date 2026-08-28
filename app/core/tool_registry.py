from typing import List, Dict, Any

# Tool definition schemas passed to LLM dialogue engine for function calling
CALL_TRANSFER_TOOL = {
    "type": "function",
    "function": {
        "name": "transfer_call",
        "description": "Transfer or forward the ongoing phone call to a real human customer care representative or target phone number when requested by the customer or required by policy.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_phone_number": {
                    "type": "string",
                    "description": "The destination phone number to transfer the call to (e.g., '+18005550199')."
                },
                "reason": {
                    "type": "string",
                    "description": "The reason for the call transfer (e.g., 'Customer requested human agent', 'Complex complaint')."
                }
            },
            "required": ["target_phone_number", "reason"]
        }
    }
}

def get_default_tools() -> List[Dict[str, Any]]:
    return [CALL_TRANSFER_TOOL]
