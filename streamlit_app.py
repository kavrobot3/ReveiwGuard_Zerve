from huggingface_hub import hf_hub_download
import joblib

st.write("Loading models from Hugging Face...")

# Use your actual model repo name here
ai_detector_path = hf_hub_download(repo_id="kavrobot/reviewguard-models", filename="ai_detector_compatible.pkl")
regret_predictor_path = hf_hub_download(repo_id="kavrobot/reviewguard-models", filename="regret_predictor_compatible.pkl")

ai_detector = joblib.load(ai_detector_path)
regret_predictor = joblib.load(regret_predictor_path)

st.success("Models loaded!")
