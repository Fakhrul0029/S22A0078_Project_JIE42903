# University Exam Scheduling using Simulated Annealing and Streamlit

import streamlit as st
import pandas as pd
import random
import math
import os
import time

# ==============================
# Page Configuration
# ==============================
st.set_page_config(page_title="Exam Scheduling with Simulated Annealing", layout="wide")
st.title("📘 University Exam Scheduling using Simulated Annealing")

st.write(
    "This application optimizes university exam scheduling by assigning exams "
    "to suitable classrooms using the Simulated Annealing algorithm."
)

# ==============================
# Load Datasets
# ==============================
st.subheader("📂 Exam and Classroom Datasets")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

exam_file = os.path.join(BASE_DIR, "exam_timeslot.csv")
room_file = os.path.join(BASE_DIR, "classrooms.csv")

if os.path.exists(exam_file):
    exams_df = pd.read_csv(exam_file)
    st.success("Exam dataset loaded successfully!")
    st.dataframe(exams_df)
else:
    st.error("Exam dataset not found.")
    st.stop()

if os.path.exists(room_file):
    rooms_df = pd.read_csv(room_file)
    st.success("Classroom dataset loaded successfully!")
    st.dataframe(rooms_df)
else:
    st.error("Classroom dataset not found.")
    st.stop()

# ==============================
# Prepare Data
# ==============================
exam_ids = exams_df["exam_id"].tolist()
room_ids = rooms_df["classroom_id"].tolist()

exam_students = dict(zip(exams_df["exam_id"], exams_df["num_students"]))
room_capacity = dict(zip(rooms_df["classroom_id"], rooms_df["capacity"]))

# ==============================
# Cost Function (with metrics)
# ==============================
def cost_function(schedule, alpha, beta):
    """
    Combined objective function:
    - Hard constraint: capacity violation
    - Soft constraint: wasted classroom capacity
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
    return total_cost, capacity_violation, wasted_capacity

# ==============================
# SA Helper Functions
# ==============================
def generate_initial_solution():
    return {exam: random.choice(room_ids) for exam in exam_ids}

def generate_neighbor(solution):
    neighbor = solution.copy()
    exam_to_change = random.choice(exam_ids)
    neighbor[exam_to_change] = random.choice(room_ids)
    return neighbor

# ==============================
# Simulated Annealing Algorithm
# ==============================
def simulated_annealing(
    initial_temp, cooling_rate, min_temp, alpha, beta, max_iter
):
    start_time = time.time()

    current_solution = generate_initial_solution()
    current_cost, _, _ = cost_function(current_solution, alpha, beta)

    best_solution = current_solution
    best_cost = current_cost

    temperature = initial_temp
    cost_history = []

    for iteration in range(max_iter):
        neighbor = generate_neighbor(current_solution)
        neighbor_cost, _, _ = cost_function(neighbor, alpha, beta)
        delta = neighbor_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_solution = neighbor
            current_cost = neighbor_cost

            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

        cost_history.append(best_cost)
        temperature *= cooling_rate

        if temperature < min_temp:
            break

    elapsed_time = time.time() - start_time
    return best_solution, best_cost, cost_history, elapsed_time

# ==============================
# Parameter Settings
# ==============================
st.subheader("⚙️ Simulated Annealing Parameters")

col1, col2 = st.columns(2)

with col1:
    initial_temp = st.slider("Initial Temperature", 500, 3000, 1500, step=100)
    cooling_rate = st.slider("Cooling Rate", 0.85, 0.99, 0.95, step=0.01)
    min_temp = st.number_input("Minimum Temperature", 1, 50, 1)

with col2:
    alpha = st.slider("Capacity Violation Weight (α)", 10, 100, 50)
    beta = st.slider("Wasted Capacity Weight (β)", 1, 20, 5)
    max_iter = st.number_input("Maximum Iterations", 50, 500, 200)

# ==============================
# Run Simulated Annealing
# ==============================
if st.button("🚀 Run Simulated Annealing"):
    with st.spinner("Running Simulated Annealing Optimization..."):
        best_schedule, best_cost, history, elapsed_time = simulated_annealing(
            initial_temp, cooling_rate, min_temp, alpha, beta, max_iter
        )

    final_cost, final_violation, final_wasted = cost_function(
        best_schedule, alpha, beta
    )

    # ==============================
    # Performance Metrics
    # ==============================
    st.subheader("📊 Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Cost", round(final_cost, 2))
    col2.metric("Capacity Violation", final_violation)
    col3.metric("Wasted Capacity", final_wasted)
    col4.metric("Computation Time (s)", f"{elapsed_time:.2f}")

    # ==============================
    # Final Schedule
    # ==============================
    st.subheader("🗓️ Final Exam Schedule")

    result_df = pd.DataFrame([
        {
            "Exam ID": e,
            "Students": exam_students[e],
            "Classroom": r,
            "Room Capacity": room_capacity[r]
        }
        for e, r in best_schedule.items()
    ])

    st.dataframe(result_df, use_container_width=True)

    output_csv = os.path.join(BASE_DIR, "sa_exam_schedule.csv")
    result_df.to_csv(output_csv, index=False)
    st.success(f"Final schedule saved to {output_csv}")

    # ==============================
    # Convergence Curve
    # ==============================
    st.subheader("📈 Convergence Curve")
    st.line_chart(history)

# ==============================
# Footer
# ==============================
st.info(
    "This project demonstrates university exam scheduling using Simulated Annealing "
    "by optimizing a combined objective of minimizing room capacity violations "
    "and wasted classroom capacity."
)
