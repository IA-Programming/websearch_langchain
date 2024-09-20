import streamlit as st
import json
import uuid
import os
from hashlib import md5
from streamlit.components.v1 import html
from streamlit_extras.add_vertical_space import add_vertical_space

def add_text(text: str) -> str:
    doc_hash = md5()
    doc_hash.update(text.encode())
    doc_id = doc_hash.hexdigest()

    return doc_id

def validate_json_content(data):
    required_keys = ['project_id', 'private_key', 'client_email']
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        return False, f"The following keys are missing in the JSON file: {', '.join(missing_keys)}"
    else:
        return True, "JSON file contains all necessary elements"

def save_validated_credentials(data: dict):
    # Define the file path for the new JSON file
    json_file_name = f'{str(uuid.uuid4())}.json'
    os.makedirs("credentials", exist_ok=True)
    output_path = os.path.join(os.getcwd(), "credentials", json_file_name)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=2)

    return output_path

def delete_json_file(json_file_name):
    try:
        os.remove(json_file_name)
        return f"{json_file_name} has been deleted!"
    except FileNotFoundError:
        return f"{json_file_name} does not exist."

def length_validation(text: str, length: int) -> bool:
    if isinstance(text, str) and len(text)==length:
        return True
    else:
        return False

def set_vertex_ai_credentials():
    google_cse_id = st.text_input(label="Google CSE ID", max_chars=17, help="ID from your custom search engine", disabled=True if st.session_state.get('google_cse_id', False) else False)
    if len(google_cse_id) and not length_validation(google_cse_id, 17): st.warning(body="review your google custom search engine id", icon="⚠️")
    google_api_key = st.text_input(label="Google API KEY", max_chars=39, type='password', help="Here the api key of your search engine", disabled=True if st.session_state.get('google_api_key', False) else False)
    if len(google_api_key) and not length_validation(google_api_key, 39): st.warning(body="review your api key", icon="⚠️")
    if length_validation(google_cse_id, 17) and length_validation(google_api_key, 39):
        st.session_state['google_cse_id'] = google_cse_id
        st.session_state['google_api_key'] = google_api_key
        st.success(body="You successfully have setup all the necessary variables for the engine", icon="✅")
    upload_file_cont = st.empty()
    uploaded_file = upload_file_cont.file_uploader("Upload your JSON file credentials", type=["json"], disabled=st.session_state.get("is_valid", False))
    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        json_string = bytes_data.decode('utf-8')
        json_dict = json.loads(json_string)
        # Validate the JSON data
        is_valid, message = validate_json_content(json_dict)
        if is_valid:
            st.session_state['is_valid'] = is_valid
            st.session_state['message'] = message
            # Save the loaded JSON under a new filename
            st.session_state['json_dict'] = json_dict
        else:
            upload_file_cont.empty()
            st.rerun()
    if st.session_state.get('is_valid', False) and isinstance(st.session_state.get('json_dict', None), dict): st.success(body=f"Validation successful: {st.session_state.get('message')}", icon="✅")
    elif 'message' in st.session_state: st.error(f"Validation failed: {message}", icon='⚠️')

    start_session = st.button(label="Start Session", help="Here when you want to start your session")
    if start_session:
        if st.session_state.get('google_cse_id', False) and st.session_state.get('google_api_key', False) and st.session_state.get('is_valid', False) and st.session_state.get('json_dict', False):
            os.environ["GOOGLE_CSE_ID"] = st.session_state.google_cse_id
            os.environ['GOOGLE_API_KEY'] = st.session_state.google_api_key
            ruta = save_validated_credentials(data=st.session_state.json_dict)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ruta
            os.environ['VERTEXAI_PROJECT'] = st.session_state.json_dict['project_id']
            st.session_state.ruta_saved = ruta
            st.session_state.session_started = True
            # Deleting all the extra variables used
            del st.session_state.google_cse_id
            del st.session_state.google_api_key
            del st.session_state.is_valid
            del st.session_state.message
            del st.session_state.json_dict
            st.rerun()


def delete_session():
    close_session = st.button("Close Session", help="This button help you delete all your variables")
    # Create the Streamlit button to delete the JSON file
    if close_session:
        st.session_state.mensaje = delete_json_file(json_file_name=st.session_state.ruta_saved)
        os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS')
        os.environ.pop('VERTEXAI_PROJECT')
        if 'GOOGLE_CSE_ID' in os.environ and 'GOOGLE_API_KEY' in os.environ:
            os.environ.pop('GOOGLE_CSE_ID')
            os.environ.pop('GOOGLE_API_KEY')
        st.session_state.session_started = False
        del st.session_state.ruta_saved
        if st.session_state.get('qa_chain', False):
            del st.session_state.session_kwargs
            del st.session_state.qa_chain
        st.rerun()

# About Us Section
def about_us():
    add_vertical_space(1)
    html_chat = '<center><h5>🤗 Support the project with a donation for the development of new Features 🤗</h5>'
    st.markdown(html_chat, unsafe_allow_html=True)
    button = '<script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="blazzmocompany" data-color="#FFDD00" data-emoji=""  data-font="Cookie" data-text="Buy me a coffee" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#ffffff" ></script>'
    html(button, height=70, width=220)
    iframe = '<style>iframe[width="220"]{position: absolute; top: 50%;left: 50%;transform: translate(-50%, -50%);margin:26px 0}</style>'
    st.markdown(iframe, unsafe_allow_html=True)
    add_vertical_space(2)
    st.write('<center><h6>Made with ❤️ by <a href="mailto:blazzmo.company@gmail.com">BlazzByte</a></h6>',
             unsafe_allow_html=True)