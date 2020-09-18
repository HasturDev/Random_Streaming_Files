from random 
from flask import Flask, jsonify

app = Flask(__name__)

class dice:
    def roll(r):
        return roll = random.randrange(r)

    def rereroll(d):
        if roll_dice > d:
    
    def remove_less(l):
        if roll_dice < l

dice.roll(20)


@app.route('/')
def roll_the_dice():
    return jsonify(
        {'roll': roll()}
    )


if __name__ == '__main__':
    app.run(threaded=True, port=5000)
