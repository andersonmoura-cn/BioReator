import sympy as sp

# Apenas para tratamento de visualização
def clean(expr, tol=1e-10, ndigits=5):
    # zera termos muito pequenos
    expr = expr.xreplace({
        n: sp.Integer(0)
        for n in expr.atoms(sp.Number)
        if abs(complex(sp.N(n))) < tol
    })

    # arredonda floats
    expr = expr.xreplace({
        n: sp.Float(round(float(sp.N(n)), ndigits))
        for n in expr.atoms(sp.Number)
        if n != 0 and n.is_real
    })

    # converte floats que são inteiros p/ Integer
    expr = expr.xreplace({
        n: sp.Integer(int(n))
        for n in expr.atoms(sp.Float)
        if float(n).is_integer()
    })

    # remove potências tipo ^1.0
    expr = expr.replace(
        lambda x: isinstance(x, sp.Pow) and x.exp == 1,
        lambda x: x.base
    )

    return sp.simplify(expr)


# alterar forma tf para array
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

