import subprocess
import sys

models = [
    "original",
    "extended_doublemlploss",
]

users = [
    10377,
    13201,
    804,
    26,
    5812,
    1,
    10801,
]

items = [
    2165,
    933,
    985,
    1047,
    571,
    2864,
    1776,
    248,
    208,
    1943,
]


import sys
import plot_user_item_curves_old as plot_script
import plot_user_item_curves_old_451 as plot_script_451

models = [
    "extended_doublemlploss",
    "original",
]

# u = 10377
# i = 933

pairs = [
    (804, 571),
    (10377, 2165),
    (5812, 248),
    (10377, 933),
    (451, 1)
]



for model in models:
    for user, item in pairs:
        print(f"\nRunning model={model}, user={user}, item={item}")

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                # "plot_user_item_curves_old.py",
                "plot_user_item_curves_old_451.py",
                "--model", model,
                "--user", str(user),
                "--item", str(item),
            ]

            plot_script_451.main()

        except Exception as exc:
            print(
                f"Skipping failed combination: "
                f"model={model}, user={user}, item={item}: {exc}"
            )

        finally:
            sys.argv = old_argv

        print(f"\nRunning model={model}, user={user}, item={item}")

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "plot_user_item_curves_old.py",
                # "plot_user_item_curves_old_451.py",
                "--model", model,
                "--user", str(user),
                "--item", str(item),
            ]

            plot_script.main()

        except Exception as exc:
            print(
                f"Skipping failed combination: "
                f"model={model}, user={user}, item={item}: {exc}"
            )

        finally:
            sys.argv = old_argv