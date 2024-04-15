from modelscope import AutoModelForCausalLM, AutoTokenizer
import streamlit as st

st.title('Qwen1.5-72B-Chat')

if 'messages' not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    system_prompt: str = st.text_area('System Prompt', value = '你是一名专业的智能助理，能够回答和解决用户提出的一切问题。')
    max_tokens: int = st.slider('Max Tokens', 1, 32000, 32000, step = 1)
    temperature: float = st.slider('Temperature', 0.1, 1.0, 0.7, step = 0.1)
    top_p: float = st.slider('Top_P', 0.1, 1.0, 1.0, step = 0.1)

    if st.button('New Chat'):
        st.session_state.messages = []
        st.experimental_rerun()

device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen1.5-72B-Chat",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen1.5-72B-Chat")

user_msg = st.chat_input('请输入...')
if user_msg:
    st.session_state.messages.append({'role': 'user', 'content': user_msg})

    prompt = user_msg
    messages = [{'role': 'system', 'content': system_prompt}] + st.session_state.messages
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_tokens = max_tokens,
        temperature = temperature,
        top_p = top_p,
        max_new_tokens = 32000
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    st.session_state.messages.append({'role': 'assistant', 'content': response})

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'], unsafe_allow_html = True)