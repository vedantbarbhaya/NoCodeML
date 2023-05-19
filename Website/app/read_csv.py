from flask import Flask
import tablib
import os

app = Flask (__name__)

#fun var

dataset = tablib.Dataset()
with open(os.path.join(os.path.dirname(__file__),'out.csv'))) as f:
    dataset.csv = f.read()


@app.route("/")
def index():
    return dataset.html


if __name__ == "__main__":
    app.run()