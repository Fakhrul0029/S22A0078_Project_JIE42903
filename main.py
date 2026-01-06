# University Exam Scheduling using Simulated Annealing and Streamlit Interface
# Multi-Objective Optimization (Rubric-Compliant)

import streamlit as st
import pandas as pd
import random
import math

st.title("University Exam Scheduling using Simulated Annealing")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
st.subheader("Exam and Classroom Dataset")

exam_file = "data/exam_timeslot.csv"
room_file = "data/classrooms.csv"

exams_df = pd.read_csv(exam_file)
rooms_df = pd.read_csv(room_file)

st.success("Datasets loaded successfully")

st.markdown("### Exam Dataset")
st.dataframe(exams_df)

st.markdown("### Classroom Dataset")
st.dataframe(rooms_df)

# -------------------------------------------------
# Convert Dataset to Readable Structures
# -------------------------------------------------
exam_ids = exams_df["exam_id"].tolist()
room_ids = rooms_df["classroom_id"].tolist()

exam_students = dict(zip(exams_df["exam_id"], exams_df["num_students"]))
room_capacity = dict(zip(rooms_df["classroom_id"], rooms_df["capacity"]))

# -------------------------------------------------
# Multi-Objective Cost Function
# -------------------------------------------------
def cost_function(schedule, alpha, beta):
    """
    Multi-objective cost:
    alpha = weight for capacity violation (hard constraint)
    beta  = weight for wasted capacity (soft constraint)
    """
    capacity_violation = 0
    wasted_capacity = 0

    for exam_id, room_id in schedule.items():
        students = exam_students[exam_id]
        capacity = room_capacity[room_id]

        if capacity < students:
            capacity_violation += (students - capacity)
        else:
            wasted_capacity += (capacity - students)

    total_cost = alpha * capacity_violation + beta * wasted_capacity
    return total_cost


# -------------------------------------------------
# Generate Initial Solution
# -------------------------------------------------
def generate_initial_solution():
    return {exam_id: random.choice(room_ids) for exam_id in exam_ids}


# -------------------------------------------------
# Generate Neighbor Solution
# -------------------------------------------------
def generate_neighbor(solution):
    neighbor = solution.copy()
    exam_to_change = random.choice(exam_ids)
    neighbor[exam_to_change] = random.choice(room_ids)
    return neighbor


# -------------------------------------------------
# Simulated Annealing Algorithm
# -------------------------------------------------
def simulated_annealing(initial_temp, cooling_rate, min_temp, alpha, beta):
    current_solution = generate_initial_solution()
    current_cost = cost_function(current_solution, alpha, beta)

    best_solution = current_solution
    best_cost = current_cost

    temperature = initial_temp
    cost_history = []

    while temperature > min_temp:
        neighbor = generate_neighbor(current_solution)
        neighbor_cost = cost_function(neighbor, alpha, beta)

        delta = neighbor_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_solution = neighbor
            current_cost = neighbor_cost

            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

        cost_history.append(best_cost)
        temperature *= cooling_rate

    return best_solution, best_cost, cost_history


# -------------------------------------------------
# Streamlit Parameters
# -------------------------------------------------
st.subheader("Simulated Annealing Parameters")

generations_info = st.caption("Higher temperature & slower cooling improve exploration but increase computation.")

# Trial 1
st.markdown("### Trial 1")
temp1 = st.slider("Initial Temperature (Trial 1)", 500, 3000, 1500, step=100, key="t1")
cool1 = st.slider("Cooling Rate (Trial 1)", 0.85, 0.99, 0.95, step=0.01, key="c1")

# Trial 2
st.markdown("### Trial 2")
temp2 = st.slider("Initial Temperature (Trial 2)", 500, 3000, 1000, step=100, key="t2")
cool2 = st.slider("Cooling Rate (Trial 2)", 0.85, 0.99, 0.90, step=0.01, key="c2")

# Trial 3
st.markdown("### Trial 3")
temp3 = st.slider("Initial Temperature (Trial 3)", 500, 3000, 2000, step=100, key="t3")
cool3 = st.slider("Cooling Rate (Trial 3)", 0.85, 0.99, 0.97, step=0.01, key="c3")

min_temp = st.number_input("Minimum Temperature", 1, 50, 1)

st.subheader("Multi-Objective Weights")
alpha = st.slider("Weight for Capacity Violation (α)", 10, 100, 50)
beta = st.slider("Weight for Wasted Capacity (β)", 1, 20, 5)

# -------------------------------------------------
# Run Experiments
# -------------------------------------------------
if st.button("Run All 3 Trials"):
    trials = [(temp1, cool1), (temp2, cool2), (temp3, cool3)]

    for i, (temp, cool) in enumerate(trials, start=1):
        st.divider()
        st.markdown(f"## 🔹 Trial {i}")

        best_schedule, best_cost, cost_history = simulated_annealing(
            temp, cool, min_temp, alpha, beta
        )

        results = []
        for exam_id, room_id in best_schedule.items():
            results.append({
                "Exam ID": exam_id,
                "Students": exam_students[exam_id],
                "Classroom": room_id,
                "Capacity": room_capacity[room_id]
            })

        result_df = pd.DataFrame(results)
        st.dataframe(result_df)

        st.success(f"Best Total Cost: {best_cost}")

        st.line_chart(cost_history)

        result_df.to_csv(f"trial_{i}_exam_schedule.csv", index=False)

st.info("This experiment demonstrates multi-objective optimization by balancing constraint satisfaction and resource efficiency.")
