import json

from rich import print
from rich.prompt import Prompt

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

print("[bold red]PROYECTO 3 - REDES BAYESIANAS[/bold red]\n")
print("Integrantes:")
print("1. Juan Bermudez")
print("2. Julian Espinoza")
print("3. Julian Ramos")
print("4. Adrian Ruiz\n")

opcion = Prompt.ask("[italic blue]Seleccione el ejercicio a probar[/italic blue] ([bold green]0[/bold green] = clase, [bold green]1[/bold green] = ejercicio 1, [bold green]2[/bold green] = ejercicio 2)")

print("\n------------------\n")

if opcion == '0':
    # ----------------- Ejercicio clase ------------------
    print("[bold italic red]EJERCICIO CLASE[/bold italic red]\n")
    # Cargar modelo
    modelo = cargar_red_desde_json("red.json")

    inferencia = VariableElimination(modelo)

    # Enunciado:
    # Sabemos que hay lluvia ligera y no hay mantenimientos programados en las vías férreas. 
    # Nos gustaría saber las probabilidades de llegar a la reunión y de fallar a la reunión.

    resultado = inferencia.query(
        variables=['Appointment'],
        evidence={'Rain': 'light', 'Maintenance': 'no'}
    )

    print("Sabemos que hay lluvia ligera y no hay mantenimientos programados en las vias ferreas. Nos gustaria saber las probabilidades de llegar a la reunion y de fallar a la reunion: \n", resultado, "\n")

    # Enunciado:
    # Sabemos que el tren viene con retraso, y no está lloviendo. 
    # Nos gustaría saber las probabilidades de llegar a la reunión y de fallar a la reunión.

    resultado2 = inferencia.query(
        variables=['Appointment'],
        evidence={'Train': 'delayed', 'Rain': 'none'}
    )

    print("Sabemos que el tren viene con retraso, y no esta lloviendo. Nos gustaria saber las probabilidades de llegar a la reunion y de fallar a la reunion: \n", resultado2, "\n")
    # --------------------------------------------------------

elif opcion == '1':
    # ----------------- EJERCICIO 1 ------------------
    print("[bold italic red]EJERCICIO 1[/bold italic red]\n")
    # Cargar modelo
    modelo = cargar_red_desde_json("ejercicio1.json")
    
    inferencia = VariableElimination(modelo)

    # Enunciado:
    # ¿Cual es la probabilidad de ser admitido a una universidad dado que el examen es dificil?

    resultado = inferencia.query(
        variables=['Admission'],
        evidence={'Exam level': 'dificil'}
    )

    print("¿Cual es la probabilidad de ser admitido a una universidad dado que el examen es dificil?: \n", resultado, "\n")

    # Enunciado:
    # ¿Cual es la probabilidad de tener un puntaje alto de aptitud?

    resultado2 = inferencia.query(
        variables=['Apti. score']
    )

    print("¿Cual es la probabilidad de tener un puntaje alto de aptitud?: \n", resultado2, "\n")

    # Enunciado:
    # ¿Cual es la distribucion de la admision?

    resultado3 = inferencia.query(
        variables=['Admission']
    )

    print("¿Cual es la distribucion de la admision?: \n", resultado3, "\n")
    # --------------------------------------------------------

elif opcion == '2':
    # ----------------- EJERCICIO 2 ------------------
    print("[bold italic red]EJERCICIO 2[/bold italic red]\n")
    # Cargar modelo
    modelo = cargar_red_desde_json("ejercicio2.json")
    
    inferencia = VariableElimination(modelo)

    # Enunciado:
    # ¿Cual es la probabilidad de ser feliz dado que asiste a fiestas frecuentemente y es terriblemente inteligente pero no muy creativo?

    resultado = inferencia.query(
        variables=['Happy'],
        evidence={'Party': 'si', 'Smart': 'si', 'Creative': 'no'}
    )

    print("¿Cual es la probabilidad de ser feliz dado que asiste a fiestas frecuentemente y es terriblemente inteligente pero no muy creativo?: \n", resultado, "\n")

    # Enunciado:
    # ¿Cual es la probabilidad de ser feliz dado que no asiste a fiestas, tiene buenas calificaciones en sus tareas y aprobo el proyecto del curso?

    resultado2 = inferencia.query(
        variables=['Happy'],
        evidence={'Party': 'no', 'HW': 'si', 'Project': 'si'}
    )

    print("¿Cual es la probabilidad de ser feliz dado que no asiste a fiestas, tiene buenas calificaciones en sus tareas y aprobo el proyecto del curso?: \n", resultado2, "\n")

    # Enunciado:
    # ¿Cual es la distribucion de la felicidad?

    resultado3 = inferencia.query(
        variables=['Happy']
    )

    print("¿Cual es la distribucion de la felicidad?: \n", resultado3, "\n")
    # --------------------------------------------------------
else:
    print("Opción no válida. Seleccione 0, 1 o 2.")