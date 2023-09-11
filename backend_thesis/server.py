from flask import Flask, jsonify
import json
import os

app = Flask(__name__)

@app.route('/thesis')
def get_json_data():
    try:
        with open('Feuille Noir.json') as file:
            data = json.load(file)
        return jsonify(data)
    except FileNotFoundError:
        return "JSON file not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == '__main__':
    app.run()
    
    
# app = Flask(__name__)

# @app. route("/members")
# def members():
#  return {"members": ["member 1", "member 2", "member 3"]}


# if __name__ == "__main__":
#     app.run(debug = true)
