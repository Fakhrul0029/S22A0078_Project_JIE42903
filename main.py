# University Exam Scheduling using Simulated Annealing and Streamlit

import streamlit as st
import pandas as pd
import random
import math
import os
import time

st.set_page_config(page_title="Exam Scheduling with Simulated Annealing", layout="wide")
st.title("University Exam Scheduling using Simulated Annealing")

# -------------------- Load Datasets --------------------
st.subheader("Exam and Classroom Dataset")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Exam dataset
exam_file = os.path.join(BASE_DIR, "exam_timeslot.csv")
if os.path.exists(exam_file):
    exams_df = pd.read_csv(exam_file)
    st.success("Exam dataset loaded successfully!")
    st.dataframe(exams_df)
else:
    st.error(f"Exam dataset not found at: {exam_file}")
    st.stop()

# Classroom dataset
room_file = os.path.join(BASE_DIR, "classrooms.csv")
if os.path.exists(room_file):
    rooms_df = pd.read_csv(room_file)
    st.success("Classroom dataset loaded successfully!")
    st.dataframe(rooms_df)
else:
    st.error(f"Classroom dataset not found at: {room_file}")
    st.stop()

# -------------------- Convert Datasets --------------------
exam_ids = exams_df["exam_id"].tolist()
room_ids = rooms_df["classroom_id"].tolist()
exam_students = dict(zip(exams_df["exam_id"], exams_df["num_students"]))
room_capacity = dict(zip(rooms_df["classroom_id"], rooms_df["capacity"]))

# -------------------- Cost Function --------------------
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

# -------------------- SA Helper Functions --------------------
def generate_initial_solution():
    return {e: random.choice(room_ids) for e in exam_ids}

def generate_neighbor(solution):
    neighbor = solution.copy()
    exam_to_change = random.choice(exam_ids)
    neighbor[exam_to_change] = random.choice(room_ids)
    return neighbor

# -------------------- Simulated Annealing --------------------
def simulated_annealing(initial_temp, cooling_rate, min_temp, alpha, beta, max_iter=100):
    start_time = time.time()
    current_solution = generate_initial_solution()
    current_cost = cost_function(current_solution, alpha, beta)
    best_solution = current_solution
    best_cost = current_cost
    temperature = initial_temp
    cost_history = []

    st.write(f"Starting Simulated Annealing: Initial Cost = {current_cost}")

    for i in range(1, max_iter + 1):
        neighbor = generate_neighbor(current_solution)
        neighbor_cost = cost_function(neighbor, alpha, beta)
        delta = neighbor_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_solution = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
                st.write(f"Iteration {i}: Best Cost updated = {best_cost}")

        cost_history.append(best_cost)

        if i % (max_iter // 10) == 0:
            st.write(f"Iteration {i}/{max_iter} | Current Best Cost: {best_cost}")

        temperature *= cooling_rate

    elapsed = time.time() - start_time
    st.write(f"Simulated Annealing completed in {elapsed:.2f} seconds")
    st.write(f"Final Best Cost: {best_cost}")
    return best_solution, best_cost, cost_history

# -------------------- Streamlit Interface --------------------
st.subheader("Parameters")
col1, col2 = st.columns(2)

with col1:
    temp = st.slider("Initial Temperature", 500, 3000, 1500, step=100)
    cool = st.slider("Cooling Rate", 0.85, 0.99, 0.95, step=0.01)
    min_temp = st.number_input("Minimum Temperature", 1, 50, 1)

with col2:
    alpha = st.slider("Weight Capacity Violation (α)", 10, 100, 50)
    beta = st.slider("Weight Wasted Capacity (β)", 1, 20, 5)
    max_iter = st.number_input("Maximum Iterations", 10, 500, 100)

# -------------------- Run SA --------------------
if st.button("Run Simulated Annealing"):
    best_schedule, best_cost, history = simulated_annealing(
        temp, cool, min_temp, alpha, beta, max_iter
    )

    # Display final schedule
    result_df = pd.DataFrame([
        {
            "Exam ID": e,
            "Students": exam_students[e],
            "Classroom": r,
            "Room Capacity": room_capacity[r]
        } for e, r in best_schedule.items()
    ])
    st.subheader("Final Exam Schedule")
    st.dataframe(result_df)

    # Save final schedule
    output_csv = os.path.join(BASE_DIR, "sa_exam_schedule.csv")
    result_df.to_csv(output_csv, index=False)
    st.success(f"Final schedule saved to {output_csv}")

    # Plot convergence using Streamlit native chart
    st.subheader("Convergence Curve")
    st.line_chart(history)

st.info(
    "This project demonstrates multi-objective optimization in exam scheduling "
    "by balancing capacity violations (hard constraint) and wasted classroom capacity (soft constraint) "
    "using Simulated Annealing."
)
