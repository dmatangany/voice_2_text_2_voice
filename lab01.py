import gradio as grd
from gtts import gTTS
import whisper
from groq import Groq
from tempfile import NamedTemporaryFile
pi_key=""
# Initialize Groq API Client
grok_client = Groq(
    api_key="",
)
model = whisper.load_model("base")

def a_voice_into_text(audio_path):
    transcription = model.transcribe(audio_path)["text"]
    return transcription

def b_generate_llm_agent_response(text):
    chat_completion = grok_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": text,
            }
        ],
    temperature=0.5,
    max_completion_tokens=1024,
        model="llama-3.1-8b-instant",
    )
    return chat_completion.choices[0].message.content

def c_text_into_voice(text):
    tts = gTTS(text)
    output_audio = NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(output_audio.name)
    return output_audio.name

def agentic_chat_pipeline(audio_path):
    try:
        # Step 1: Convert speech to text
        text_input = a_voice_into_text(audio_path)

        # Step 2: Get response from LLaMA model
        response_text = b_generate_llm_agent_response(text_input)

        # Step 3: Convert response text to speech
        response_audio_path = c_text_into_voice(response_text)

        return response_text, response_audio_path

    except Exception as e:
        return str(e), None

# Create Gradio Interface
iface = grd.Interface(
    fn=agentic_chat_pipeline,
    inputs=grd.Audio(type="filepath", label="Speak"),  # Removed 'source' argument
    outputs=[
        grd.Textbox(label="Response Text"),
        grd.Audio(label="Response Audio")
    ],
    title="Real-Time Voice-to-Voice Chatbot"
)

iface.launch()

