from globals import STR_TRUE_PI

def precision_pi(calc_pi):
    precision = 0
    strCalcPi = f"{calc_pi:.20f}"
    for t, c in zip(STR_TRUE_PI, strCalcPi):
        if t == c:
            precision += 1
        else:
            break
    return precision

def colored_pi(calc):
    result = ""
    diff = False
    precision = 0
    for t, c in zip(STR_TRUE_PI, calc):
        if t == c and not diff:
            result += f"\033[92m{c}\033[0m"   # green
            precision += 1
        else:
            diff = True
            result += f"\033[0m{c}"
    if len(calc) > len(STR_TRUE_PI):
        result += f"\033[91m{calc[len(STR_TRUE_PI):]}\033[0m"
    return result, precision

