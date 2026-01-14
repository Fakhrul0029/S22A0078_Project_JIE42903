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
st.set_page_config(
    page_title="Exam Scheduling with Simulated Annealing",
    layout="wide"
)

st.title("🎓 University Exam Scheduling using Simulated Annealing")

# ==============================
# Load Datasets
# ==============================
st.subheader("📂 Exam and Classroom Dataset")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

exam_file = os.path.join(BASE_DIR, "exam_timeslot.csv")
room_file = os.path.join(BASE_DIR, "classrooms.csv")

if os.path.exists(exam_file) and os.path.exists(room_file):
    exams_df = pd.read_csv(exam_file)
    rooms_df = pd.read_csv(room_file)
    st.success("Datasets loaded successfully!")
else:
    st.error("Dataset files not found. Please check your repository.")
    st.stop()

st.dataframe(exams_df)
st.dataframe(rooms_df)

# ==============================
# Prepare Data
# ==============================
exam_ids = exams_df["exam_id"].tolist()
room_ids = rooms_df["classroom_id"].tolist()

exam_students = dict(zip(exams_df["exam_id"], exams_df["num_students"]))
room_capacity = dict(zip(rooms_df["classroom_id"], rooms_df["capacity"]))

# ==============================
# Cost Function (Multi-objective → Single combined)
# ==============================
def cost_function(schedule, alpha, beta):
    capacity_violation = 0
    wasted_capacity = 0

    for exam_id, room_id in schedule.items():
        students = exam_students[exam_id]
        capacity = room_capacity[room_id]

        if students > capacity:
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
    exam = random.choice(exam_ids)
    neighbor[exam] = random.choice(room_ids)
    return neighbor

# ==============================
# Simulated Annealing Algorithm
# ==============================
def simulated_annealing(
    initial_temp,
    cooling_rate,
    min_temp,
    alpha,
    beta,
    max_iter
):
    start_time = time.time()

    current_solution = generate_initial_solution()
    current_cost, _, _ = cost_function(current_solution, alpha, beta)

    best_solution = current_solution
    best_cost = current_cost

    temperature = initial_temp
    cost_history = []

    for iteration in range(1, max_iter + 1):
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
# Streamlit Parameters
# ==============================
st.subheader("⚙️ Simulated Annealing Parameters")

col1, col2 = st.columns(2)

with col1:
    initial_temp = st.slider(
        "Initial Temperature",
        min_value=500,
        max_value=5000,
        value=3000,
        step=100
    )

    cooling_rate = st.slider(
        "Cooling Rate",
        min_value=0.90,
        max_value=0.99,
        value=0.98,
        step=0.01
    )

    min_temp = st.number_input(
        "Minimum Temperature",
        min_value=1,
        value=1
    )

with col2:
    max_iter = st.number_input(
        "Maximum Iterations",
        min_value=10,
        value=1000,
        step=100
    )

    alpha = st.slider(
        "Capacity Violation Weight (α)",
        min_value=10,
        max_value=100,
        value=50
    )

    beta = st.slider(
        "Wasted Capacity Weight (β)",
        min_value=1,
        max_value=20,
        value=5
    )

# ==============================
# Run Simulated Annealing
# ==============================
if st.button("🚀 Run Simulated Annealing"):
    with st.spinner("Running Simulated Annealing..."):
        best_solution, best_cost, history, elapsed_time = simulated_annealing(
            initial_temp,
            cooling_rate,
            min_temp,
            alpha,
            beta,
            max_iter
        )

    final_cost, cap_violation, wasted_cap = cost_function(
        best_solution, alpha, beta
    )

    # ==============================
    # Performance Metrics
    # ==============================
    st.subheader("📊 Performance Metrics")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Cost", round(final_cost, 2))
    m2.metric("Capacity Violation", cap_violation)
    m3.metric("Wasted Capacity", wasted_cap)
    m4.metric("Computation Time (s)", round(elapsed_time, 3))

    # ==============================
    # Convergence Curve
    # ==============================
    st.subheader("📈 Convergence Curve")
    st.line_chart(history)

    # ==============================
    # Final Schedule
    # ==============================
    st.subheader("🗓️ Optimized Exam Schedule")

    result_df = pd.DataFrame([
        {
            "Exam ID": exam,
            "Students": exam_students[exam],
            "Classroom": room,
            "Room Capacity": room_capacity[room]
        }
        for exam, room in best_solution.items()
    ])

    st.dataframe(result_df, use_container_width=True)

    # Save result
    output_file = os.path.join(BASE_DIR, "sa_exam_schedule.csv")
    result_df.to_csv(output_file, index=False)
    st.success(f"Final schedule saved to {output_file}")

# ==============================
# Footer
# ==============================
st.info(
    "This project applies Simulated Annealing (SA), a probabilistic metaheuristic, "
    "to solve a university exam scheduling problem by minimizing capacity violations "
    "and wasted classroom capacity using a combined cost function."
)
