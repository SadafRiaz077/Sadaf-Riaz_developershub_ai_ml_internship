
Task 4: General Health Query Chatbot (Prompt Engineering Based)

DevelopersHub Corporation — AI/ML Engineering Internship


Task Objective

Build a chatbot that answers general health-related questions using a Large Language Model (LLM). The chatbot uses prompt engineering to produce friendly, clear, and safe responses, and includes safety filters to prevent giving harmful or emergency-level medical advice.


Dataset Used

This task does not use a traditional dataset. Instead, the chatbot is powered by a pre-trained Large Language Model:


Model: mistralai/Mistral-7B-Instruct-v0.2
Source: Hugging Face Model Hub (fully open-source, no API key or token required)
Loading method: 4-bit quantization via bitsandbytes for fast inference on free-tier Colab GPU



Models Applied

ComponentDetailsLLMMistral-7B-Instruct-v0.2 (7 billion parameters)Quantization4-bit NF4 via BitsAndBytesConfig (reduces memory ~75%)InferenceHugging Face pipeline("text-generation")Prompt FormatMistral instruct format: <s>[INST] ... [/INST]


Approach

Prompt Engineering

Every user query is wrapped in a structured persona-based prompt before being sent to the model:

<s>[INST]
Act like a helpful, friendly general health assistant.
Answer the following health question in simple, clear language for a general audience.
Do not give specific medical diagnoses, dosages, or treatment instructions.
If the question requires a doctor's attention, gently suggest consulting a healthcare professional.
Keep the answer short (3-5 sentences).

Question: {user_query}
[/INST]

This technique, known as instruction-following prompting, shapes the model's tone, length, and level of caution.

Safety Filters

A keyword-based safety layer intercepts queries before they reach the model:


Emergency filter: Queries containing words like "chest pain", "can't breathe", "suicide", "heart attack", "overdose", etc., are intercepted and the model is NOT called. Instead, the user is immediately directed to emergency services.
Dosage filter: Queries containing words like "dosage", "mg", "milligram", "how many pills" are flagged. A disclaimer is prepended to the model's answer advising the user to consult a pharmacist or doctor.



Key Results and Findings

Example Queries and Behaviour

QueryBehaviour"What causes a sore throat?"Model answers clearly in plain language"Is paracetamol safe for children?"Model answers with appropriate caution"What should I do if I have a mild fever?"Model gives general self-care advice"How can I improve my sleep quality?"Model gives lifestyle tips"I have chest pain and can't breathe"Emergency filter fires — model NOT called"How many mg of ibuprofen should I take?"️ Dosage disclaimer prepended to answer

Skills Demonstrated


Prompt design and testing — Persona-based instruction prompting using Mistral's [INST]...[/INST] format
Using APIs/LLMs — Hugging Face Mistral-7B-Instruct loaded in 4-bit for speed on free Colab GPU; no API key required
Safety handling in chatbot responses — Emergency detection redirects to services; dosage queries receive automatic disclaimers
Building simple conversational agents — Interactive chat loop with a clean user interface in the notebook



How to Run

Requirements

bashpip install transformers accelerate bitsandbytes sentencepiece

Environment


Google Colab with T4 GPU runtime (Runtime → Change runtime type → T4 GPU)
First run downloads the model (~4–5 GB in 4-bit). Subsequent runs use the cached version.


Steps


Open Task4_Health_Chatbot_Mistral7B.ipynb in Google Colab.
Switch to T4 GPU runtime.
Run cells sequentially (Steps 1–9).
For interactive testing, run Step 9 and type your own health questions.



Project Structure

Task4_Health_Chatbot_Mistral7B.ipynb ← Main notebook


Important Notes


This chatbot is for general informational purposes only and is not a substitute for professional medical advice.
The model is loaded in 4-bit quantization — this trades a negligible amount of quality for a ~75% reduction in memory usage and significantly faster load times on free GPU tiers.
If generation is still slow, you can reduce max_new_tokens from 120 to 60 in the ask_health_bot() function with no other changes needed.



References


Mistral-7B-Instruct-v0.2 on Hugging Face
Hugging Face Transformers Documentation
BitsAndBytes 4-bit Quantization
Colab Notebook link:
https://colab.research.google.com/drive/1kG3ZUm3thdfdxn0M7evRhA6i4iCPm7Cg?usp=sharing
