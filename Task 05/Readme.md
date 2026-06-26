Task 5: Mental Health Support Chatbot (Fine-Tuned)

DevelopersHub Corporation — AI/ML Engineering Internship


Task Objective

Build a basic chatbot that provides supportive and empathetic responses for stress, anxiety, and emotional wellness. The chatbot is created by fine-tuning a small LLM on real empathetic human dialogues, teaching it to respond with a gentle, emotionally supportive tone. It also includes a safety filter that detects crisis-level language and redirects users to professional help.


Dataset Used

EmpatheticDialogues — Facebook AI Research

PropertyDetailsSourcebdotloh/empathetic-dialogues-contexts (Hugging Face Datasets)Size25,000+ dialogues across 32 emotion categoriesEmotions coveredanxious, sad, lonely, afraid, angry, hopeful, caring, guilty, ashamed, terrified, and moreTraining split used3,000 samples (filtered to 15 mental-health-relevant emotion categories)Validation split used300 samples

The dataset was explored and visualised before training to understand emotion category distributions and text length distributions.


Models Applied

ComponentDetailsBase ModelEleutherAI/GPT-Neo-125M (125 million parameters)Fine-tuning methodCausal Language Modeling via Hugging Face Trainer APITraining epochs3Batch size8PrecisionFP16 on GPUPrompt formatInstruction-style: Emotion: {emotion}\nUser: {input}\nAssistant:Evaluation metricPerplexity (derived from eval loss)

GPT-Neo 125M was chosen as a lightweight, open-source base model that can be fine-tuned on a free Colab T4 GPU within a reasonable time. As noted in the summary, Mistral-7B would yield higher quality responses but requires Colab Pro.


Approach

Fine-Tuning Pipeline


Load & Explore Dataset — EmpatheticDialogues loaded via datasets library; emotion distributions and text length distributions visualised.
Preprocess — Dialogues converted to instruction-format training strings and tokenized. Samples longer than the max token length are filtered out.
Fine-tune — GPT-Neo 125M fine-tuned using Trainer with causal language modeling loss. Training and validation loss tracked across all steps.
Evaluate — Perplexity computed from validation loss. Loss curves plotted.
Save Model — Fine-tuned weights and tokenizer saved to ./gpt_neo_mental_health_final.


Response Generation

The chatbot uses instruction-style prompting combined with sampling-based decoding for diverse, natural responses:

You are a compassionate mental health support assistant.
Emotion: {emotion_hint}
User: {user_input}
Assistant:

Decoding parameters: temperature=0.6, top_p=0.88, top_k=40, repetition_penalty=1.4

Safety Filter

A keyword-based crisis filter is applied both to user input and to generated responses:


Trigger words: suicide, kill yourself, self-harm, hurt yourself, end your life, overdose
Action: If any trigger word is detected (in either input or output), the model response is replaced with a crisis message directing the user to a mental health professional or crisis helpline.
The model is never allowed to output harmful content — the safety check covers the generated text as well.



Key Results and Findings

Training Summary

MetricValueEpochs3Training samples3,000Validation samples300Evaluation metricPerplexityLoss trackingTrain loss + Validation loss per step (plotted)

Sample Test Responses

EmotionUser InputBot BehaviourAnxious"I've been feeling really anxious about my exams and can't sleep."Empathetic, supportive responseLonely"I feel so lonely. Nobody seems to understand me."Validating, warm responseSad"I'm overwhelmed with work and feel like I can't cope anymore."Gentle, caring responseHopeful"I finally finished a big project — feeling hopeful!"Celebratory, encouraging responseCrisisAny message with self-harm keywords️ Safety filter fires — crisis helpline message returned

Key Findings


Fine-tuning on domain-specific empathetic data meaningfully improves the model's emotional tone compared to the base GPT-Neo.
Instruction-style prompts (with an emotion hint) guide response quality and keep answers relevant.
GPT-Neo 125M outperforms DistilGPT2 for conversational empathetic tasks due to its larger capacity.
Safety filters are essential for any mental health chatbot — the model alone cannot be trusted to never produce harmful output.
Mistral-7B would produce higher-quality, more coherent responses; GPT-Neo 125M is a practical choice for free-tier GPU constraints.


Skills Demonstrated


Model fine-tuning with Hugging Face Transformers — Full fine-tuning pipeline using Trainer API with DataCollatorForLanguageModeling
Emotional tone design for safe chatbots — Instruction prompts with emotion hints; safety layer covering both input and output
Conversation modeling — Causal LM fine-tuning on real empathetic human dialogues
Deployment using CLI or simple web apps — Interactive chat loop in Colab notebook with emotion-hint interface



How to Run

Requirements

bashpip install transformers datasets accelerate

Environment


Google Colab with T4 GPU runtime (Runtime → Change runtime type → T4 GPU)


Steps


Open Task5_Mental_Health_Chatbot.ipynb in Google Colab.
Switch to T4 GPU runtime.
Run all cells sequentially (Steps 1–12).
For interactive testing, run Step 11 (Interactive Chat Loop).



Project Structure

Task5_Mental_Health_Chatbot.ipynb ← Main notebook
gpt_neo_mental_health_final/ ← Saved fine-tuned model (created after training)
emotion_distribution.png ← Dataset visualisation
training_loss.png ← Training & validation loss curves


Important Notes


This chatbot is not a replacement for professional mental health support. It is an educational project demonstrating fine-tuning techniques.
If you or someone you know is in crisis, please contact a qualified mental health professional or a crisis helpline immediately.
Training takes approximately 10–20 minutes on a free Colab T4 GPU with 3,000 samples.



References


EmpatheticDialogues Dataset (Facebook AI)
EleutherAI GPT-Neo on Hugging Face
Hugging Face Trainer API Documentation
