import json
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def cargar_red_desde_json(ruta):
    with open(ruta, 'r') as f:
        data = json.load(f)

    model = DiscreteBayesianNetwork(data['structure'])
    variables = data['variables']

    for prob_cond_tab_info in data['prob_cond_tabs']:
        var = prob_cond_tab_info['variable']
        states = variables[var]
        values = prob_cond_tab_info['probabilidades']

        if 'evidencia' in prob_cond_tab_info:
            evidence = prob_cond_tab_info['evidencia']
            evidence_card = [len(variables[e]) for e in evidence]
            state_names = {var: states}
            state_names.update({e: variables[e] for e in evidence})
        else:
            evidence = None
            evidence_card = None
            state_names = {var: states}

        prob_cond_tab = TabularCPD(
            variable=var,
            variable_card=len(states),
            values=values,
            evidence=evidence,
            evidence_card=evidence_card,
            state_names=state_names
        )

        model.add_cpds(prob_cond_tab)

    if model.check_model():
        return model
    else:
        raise ValueError("Modelo inválido")

# Cargar modelo
modelo = cargar_red_desde_json("red.json")

# ----------------- Realizar inferencia ------------------
inferencia = VariableElimination(modelo)

# Enunciado:
# Sabemos que hay lluvia ligera y no hay mantenimientos programados en las vías férreas. 
# Nos gustaría saber las probabilidades de llegar a la reunión y de fallar a la reunión.

resultado = inferencia.query(
    variables=['Appointment'],
    evidence={'Rain': 'light', 'Maintenance': 'no'}
)

print("Query 1: \n", resultado, "\n")

# Enunciado:
# Sabemos que el tren viene con retraso, y no está lloviendo. 
# Nos gustaría saber las probabilidades de llegar a la reunión y de fallar a la reunión.

resultado2 = inferencia.query(
    variables=['Appointment'],
    evidence={'Train': 'delayed', 'Rain': 'none'}
)

print("Query 2: \n", resultado2, "\n")
# --------------------------------------------------------
