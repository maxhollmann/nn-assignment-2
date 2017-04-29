def log(s, level = 'debug'):
    with open("out/results.txt", "a") as f:
        f.write(str(s) + "\n")
    print(s)
