import argparse
from pushover import Pushover


def main():
    parser = argparse.ArgumentParser(
        description='Sends a message to app to notify')

    parser.add_argument('--msg', type=str, required=True,
                        help='The message to send')
    parser.add_argument('--title', type=str, required=True, help='The title of the message')
    parser.add_argument('--app_token', type=str, required=True, help='The token for the app')
    parser.add_argument('--user_token', type=str, required=True, help='The token for the user')

    args = parser.parse_args()


    po = Pushover(args.app_token)
    po.user(args.user_token)

    msg = po.msg(args.msg)

    msg.set("title", args.title)

    po.send(msg)


if __name__ == "__main__":
    main()