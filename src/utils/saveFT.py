import sympy as sp

def tf_to_num_den(expr, var):
    num, den = sp.fraction(sp.simplify(expr))
    
    num_poly = sp.Poly(num, var)
    den_poly = sp.Poly(den, var)
    
    num_coeffs = num_poly.all_coeffs()
    den_coeffs = den_poly.all_coeffs()
    
    # converter para float (opcional)
    num_coeffs = [float(c) for c in num_coeffs]
    den_coeffs = [float(c) for c in den_coeffs]
    
    return num_coeffs, den_coeffs


def save_mimo_tf(G, var, filename="config/tf_config.py"):
    outputs = ["X", "S"]
    inputs = ["D", "Sf"]

    with open(filename, "w") as f:
        # f.write("# Funcoes de transferencia MIMO\n\n")
        
        for i, y in enumerate(outputs):
            for j, u in enumerate(inputs):
                
                num, den = tf_to_num_den(G[i, j], var)
                
                name = f"G_{y}_{u}"
                
                f.write(f"{name} = {{\n")
                f.write(f"    'num': {num},\n")
                f.write(f"    'den': {den}\n")
                f.write("}\n\n")