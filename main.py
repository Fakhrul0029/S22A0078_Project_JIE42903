# University Exam Scheduling using Simulated Annealing and Streamlit Interface

import streamlit as st
import pandas as pd
import random
import math
import os

st.title("University Exam Scheduling using Simulated Annealing")

# -------------------------------------------------
# Load Dataset (Safe Version)
# -------------------------------------------------
st.subheader("Exam and Classroom Dataset")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Exam dataset
exam_file_path = os.path.join(BASE_DIR, "data", "exam_timeslot.csv")
if os.path.exists(exam_file_path):
    exams_df = pd.read_csv(exam_file_path)
    st.success("Exam dataset loaded successfully!")
    st.dataframe(exams_df)
else:
    st.error(f"Exam dataset not found at: {exam_file_path}")
    st.stop()  # Stop Streamlit execution if file is missing

# Classroom dataset
classroom_file_path = os.path.join(BASE_DIR, "data", "classrooms.csv")
if os.path.exists(classroom_file_path):
    rooms_df = pd.read_csv(classroom_file_path)
    st.success("Classroom dataset loaded successfully!")
    st.dataframe(rooms_df)
else:
    st.error(f"Classroom dataset not found at: {classroom_file_path}")
    st.stop()  # Stop Streamlit execution if file is missing

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
    Multi-objective cost function:
    - Capacity violation (hard constraint)
    - Wasted capacity (soft constraint)
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
    solution = {}
    for exam_id in exam_ids:
        solution[exam_id] = random.choice(room_ids)
    return solution

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
# Streamlit Interface (Parameters)
# -------------------------------------------------
st.subheader("Simulated Annealing Parameters")

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

# Multi-objective weights
st.subheader("Multi-Objective Weights")
alpha = st.slider("Weight for Capacity Violation (α)", 10, 100, 50)
beta = st.slider("Weight for Wasted Capacity (β)", 1, 20, 5)

# -------------------------------------------------
# Run and Display Results
# -------------------------------------------------
if st.button("Run All 3 Trials"):
    trials = [
        (temp1, cool1),
        (temp2, cool2),
        (temp3, cool3)
    ]

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
                "Room Capacity": room_capacity[room_id]
            })

        result_df = pd.DataFrame(results)
        st.dataframe(result_df)

        st.success(f"Best Total Cost (Lower is Better): {best_cost}")
        st.line_chart(cost_history)

        # Save result for documentation
        output_file = os.path.join(BASE_DIR, f"trial_{i}_exam_schedule.csv")
        result_df.to_csv(output_file, index=False)
        st.info(f"Trial {i} results saved to {output_file}")

st.info(
    "This project demonstrates multi-objective optimization in exam scheduling "
    "by balancing constraint satisfaction and efficient classroom utilization "
    "using Simulated Annealing."
)
