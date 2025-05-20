# 🧠 NILE - NeuroSymbolic Interactive Language Engine (Prototype)

> A hybrid prototype language engine combining BERT-based Japanese language understanding with symbolic reasoning.

---

## 📌 Overview

**NILE** is an experimental prototype designed to explore a new kind of language processing system that:
- Uses **BERT-based Japanese Model** for natural language understanding
- Leverages a **Symbolic Reasoning Engine** for logic-based inference
- Provides **explainable responses** with traceable reasoning steps

---

## 🚀 Features

- ✅ Natural language fact registration (e.g. "A is a kind of B")
- ✅ Logic-based inference (e.g. "Is A a kind of C?")
- ✅ Explainable reasoning in human-readable language
- ✅ Simple CLI interface for testing and demonstration

---



## ⚙️ Requirements

- Python 3.8+
- Required packages:
  ```bash
  pip install transformers torch python-dotenv sympy protobuf fugashi
  ```

## 💡 Example Usage

> python main.py
Input: A is a kind of B.
[Symbolic Engine] Fact stored.

Input: B is a kind of C.
[Symbolic Engine] Fact stored.

Input: Is A a kind of C?
[Symbolic Engine] Reasoning: A -> B -> C
Output: Yes. Because A is a kind of B, and B is a kind of C, therefore A is a kind of C.

## 🔍 Pattern Matching

The system recognizes the following patterns:

1. Question patterns:
   - "What is ~?"
   - "Tell me about ~"
   - "What is ~?"
   - "What are the characteristics of ~?"
   - "What is the state of ~?"

2. Fact patterns:
   - "~ is ~"
   - "~ is ~"
   - "~ is ~"
   - "~ is a type of ~"
   - "~ belongs to ~"

## 🔄 Relationship Inference

The system infers the following relationships:

1. Basic relationships:
   - Type (is_a)
   - Property (has_property)
   - Related (related_to)

2. Time-related:
   - Time (time)
   - State (state)

## 🚧 Future Improvements

1. Pattern matching enhancement
   - Support for more Japanese expressions
   - Context-based relationship inference

2. Error handling improvement
   - More detailed error messages
   - Recovery process implementation

3. Performance optimization
   - Index efficiency
   - Search speed improvement

## 📝 License

MIT License

