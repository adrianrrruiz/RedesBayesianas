from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Cada tupla indica una relación de ("Padre", "Hijo")
modelo = DiscreteBayesianNetwork([
    ("Rain", "Maintenance"),
    ("Rain", "Train"),
    ("Maintenance", "Train"),
    ("Train", "Appointment")
])

# --------- Inicilizar tablas de probabilidad condicional ---------
prob_cond_tab_rain = TabularCPD(
    variable='Rain',
    variable_card=3,
    values=[[0.7], [0.2], [0.1]],
    state_names={'Rain': ['none', 'light', 'heavy']}
)

prob_cond_tab_maintenance = TabularCPD(
    variable='Maintenance',
    variable_card=2,
    values=[
        [0.4, 0.2, 0.1], #yes
        [0.6, 0.8, 0.9] #no
    ],
    evidence=['Rain'],
    evidence_card=[3],
    state_names={
        'Maintenance': ['yes', 'no'],
        'Rain': ['none', 'light', 'heavy']
    }
)

prob_cond_tab_train = TabularCPD(
    variable='Train',
    variable_card=2,
    values=[
        [0.8, 0.9, 0.6, 0.7, 0.4, 0.5], #on time
        [0.2, 0.1, 0.4, 0.3, 0.6, 0.5] #delayed
    ],
    evidence=['Rain', 'Maintenance'],
    evidence_card=[3, 2],
    state_names={
        'Train': ['on time', 'delayed'],
        'Rain': ['none', 'light', 'heavy'],
        'Maintenance': ['yes', 'no']
    }
)

prob_cond_tab_appointment = TabularCPD(
    variable='Appointment',
    variable_card=2,
    values=[
        [0.9, 0.6], #attend
        [0.1, 0.4] #miss
    ],
    evidence=['Train'],
    evidence_card=[2],
    state_names={
        'Appointment': ['attend', 'miss'],
        'Train': ['on time', 'delayed']
    }
)

modelo.add_cpds(
    prob_cond_tab_rain,
    prob_cond_tab_maintenance,
    prob_cond_tab_train,
    prob_cond_tab_appointment
)
# --------------------------------------------------------

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
