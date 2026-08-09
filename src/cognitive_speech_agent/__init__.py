"""Cognitive Speech Screening Agent.

Research demo only. Not a medical device. Clinician-in-the-loop.

Pipeline (the five tools):
    1. diarize   -> split speakers
    2. transcribe-> speech to text with word timings
    3. markers   -> compute cognitive-linguistic features
    4. scoring   -> markers to risk score + norm flags
    5. report    -> grounded LLM writes the screening report
The orchestrator wires them together as an agent loop.
"""

__version__ = "0.1.0"
