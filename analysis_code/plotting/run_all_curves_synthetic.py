import sys
import plot_user_item_curves_synthetic as plot_script

models = [
    "extended_doublemlploss",
    "original",
]

pairs = [
    (50, 100)
]

for model in models:
    for user, item in pairs:
        print(f"\nRunning model={model}, user={user}, item={item}")

        old_argv = sys.argv[:]
        # try:
        sys.argv = [
            "plot_user_item_curves_synthetic.py",
            "--model", model,
            "--user", str(user),
            "--item", str(item),
        ]

        plot_script.main()

        # except Exception as exc:
        #     print(
        #         f"Skipping failed combination: "
        #         f"model={model}, user={user}, item={item}: {exc}"
        #     )

        # finally:
        #     sys.argv = old_argv