import subprocess

attacks = [
    ("none", None),
    # ("label_flip", None),
    # ("attribute_flip", None),
    # ("double_flip", None),
    # ("hybrid_flip", None),
    # ("concealed", "data"),
    # ("concealed", "model")
]
# defenses = ["fairfed"]
# defenses = ["decaf"]
defenses = ['fedavg', 'fairfate', 'fairfed', 'fairguard', 'fltrust', 'krum']

for attack, concealed_mode in attacks:
    for mitigation in defenses:
        print("=" * 80)
        print(f"▶️ Running experiment: ATTACK = {attack} | MODE = {concealed_mode} | DEFENSE = {mitigation}")
        print("=" * 80)

        cmd = [
            "python", "federated_framework.py",
            "--dataset", "adult",
            "--sensitive_attr", "race",
            "--model", "fcnn",
            "--attack", attack,
            "--mitigation", mitigation,
            "--rounds", "40",
            "--num_clients", "50",
            "--target_group","1",
            "--num_class", "2"
        ]

        if concealed_mode:
            cmd += ["--concealed_mode", concealed_mode]

        subprocess.run(cmd)
