#!/usr/bin/env python3
"""
Test to demonstrate context-aware intent classification.
"""

def simulate_conversation_flow():
    """Simulate how the orchestrator would now handle context-aware intent classification."""
    
    print("🤖 Context-Aware Intent Classification Demo")
    print("=" * 50)
    
    # Simulate conversation flow
    conversations = [
        {
            "messages": [
                {"role": "user", "content": "write python code to reverse a string"},
                {"role": "assistant", "content": "Here's a Python function to reverse a string:\n\n```python\ndef reverse_string(s):\n    return s[::-1]\n```"},
                {"role": "user", "content": "explain why"}
            ],
            "expected_flow": "Code question → Code response → Follow-up about code (should be 'code' intent)"
        },
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
                {"role": "user", "content": "explain why"}
            ],
            "expected_flow": "General question → General response → Follow-up about general topic (should be 'general' intent)"
        },
        {
            "messages": [
                {"role": "user", "content": "Write a poem about the ocean"},
                {"role": "assistant", "content": "# Ocean's Song\n\nWaves crash upon the shore so bright,\nUnder the moon's gentle light..."},
                {"role": "user", "content": "make it shorter"}
            ],
            "expected_flow": "Creative writing → Creative response → Modification request (should be 'creative writing' intent)"
        }
    ]
    
    for i, conv in enumerate(conversations, 1):
        print(f"\n📝 Conversation {i}:")
        print(f"Expected flow: {conv['expected_flow']}")
        print()
        
        context = ""
        for j, msg in enumerate(conv["messages"]):
            print(f"  {msg['role'].capitalize()}: {msg['content']}")
            
            if j == len(conv["messages"]) - 1:  # Last message (the follow-up)
                print()
                print("  🧠 Orchestrator Analysis:")
                print(f"     - Current message: '{msg['content']}'")
                print(f"     - Previous context: {context[:100]}...")
                print("     - With context, can determine this relates to previous topic")
                print("     - Will use classify_intent tool with conversation_context parameter")
            else:
                context += f"{msg['role']}: {msg['content']} | "
        
        print()
    
    print("✨ Benefits of Context-Aware Classification:")
    print("  • Better handling of follow-up questions")
    print("  • Maintains conversation coherence")
    print("  • Reduces misclassification of ambiguous queries")
    print("  • Enables more intelligent routing decisions")

if __name__ == "__main__":
    simulate_conversation_flow()
