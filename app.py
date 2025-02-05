import whisper
import base64
import tempfile
from flask_sock import Sock
from flask import Flask, render_template

app = Flask(__name__)
sock= Sock(app)

app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}


model = whisper.load_model('base.en')

def process_wav_bytes(webm_bytes: bytes, sample_rate: int = 16000):
	with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
		print('!',  temp_file.name)
		temp_file.write(webm_bytes)
		temp_file.flush()
		waveform = whisper.load_audio(temp_file.name, sr=sample_rate)
		return waveform

@app.route('/')
def index():
	return render_template('index.html')

@sock.route('/transcription')
def transcribe_socket(ws):
	print(ws)
	#while not ws.closed():
	while True:
		message = ws.receive()
		if message:
			print('message received', len(message), type(message))
			if isinstance(message, str):
				message = base64.b64decode(message)
			audio = process_wav_bytes(bytes(message)).reshape(1, -1)
			print('PWB:',audio)
			audio = whisper.pad_or_trim(audio)
			print('P/T:',audio)
			transcription = whisper.transcribe(model, audio, fp16=False) # on CPU
			print('TS:', transcription)
			ws.send(transcription["text"])

if __name__ == "__main__":
	app.run()
