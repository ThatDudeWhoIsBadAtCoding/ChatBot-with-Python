from flask import Flask, render_template
from flask_socketio import SocketIO, send
import chatbot

app = Flask(__name__)
app.config["insert da key here"] = "very secret code bro"
socket = SocketIO(app, cors_allowed_origins="*")


@app.route("/")
def home():
    return render_template("index.html")


@socket.on("message")
def handle(msg):
    if msg == "User has connected!":
        return
    msg_ = msg.lower()
    send(f"You: {msg}", broadcast=True)
    ints = chatbot.predictclass(msg_)
    resp = chatbot.getdatresponse(ints, chatbot.intents)
    send(f"Bot: {resp}", broadcast=True)


if __name__ == "__main__":
    socket.run(app, host="localhost")
