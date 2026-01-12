import streamlit as st
import pandas as pd
import random
import math
import os
import time
import matplotlib.pyplot as plt

st.title("University Exam Scheduling using Simulated Annealing")

# Load datasets (assume files are in same folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
exam_file = os.path.join(BASE_DIR, "exam_timeslot.csv")
room_file = os.path.join(BASE_DIR, "classrooms.csv")

exams_df = pd.read_csv(exam_file)
rooms_df = pd.read_csv(room_file)

exam_ids = exams_df["exam_id"].tolist()
room_ids = rooms_df["classroom_id"].tolist()
exam_students = dict(zip(exams_df["exam_id"], exams_df["num_students"]))
room_capacity = dict(zip(rooms_df["classroom_id"], rooms_df["capacity"]))

# ------------------ Cost Function ------------------
def cost_function(schedule, alpha, beta):
    cap_violation = 0
    wasted = 0
    for e,r in schedule.items():
        students = exam_students[e]
        capacity = room_capacity[r]
        if capacity < students:
            cap_violation += (students - capacity)
        else:
            wasted += (capacity - students)
    return alpha*cap_violation + beta*wasted

# ------------------ SA Functions ------------------
def generate_initial_solution():
    return {e: random.choice(room_ids) for e in exam_ids}

def generate_neighbor(sol):
    n = sol.copy()
    exam_to_change = random.choice(exam_ids)
    n[exam_to_change] = random.choice(room_ids)
    return n

def simulated_annealing(initial_temp, cooling_rate, min_temp, alpha, beta, max_iter=100):
    start_time = time.time()
    current_solution = generate_initial_solution()
    current_cost = cost_function(current_solution, alpha, beta)
    best_solution = current_solution
    best_cost = current_cost
    temperature = initial_temp
    cost_history = []

    st.write(f"Starting SA: initial cost = {current_cost}")
    for i in range(1, max_iter+1):
        neighbor = generate_neighbor(current_solution)
        neighbor_cost = cost_function(neighbor, alpha, beta)
        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta/temperature):
            current_solution = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
                st.write(f"Iteration {i}: Best Cost updated = {best_cost}")
        cost_history.append(best_cost)
        temperature *= cooling_rate

    elapsed = time.time() - start_time
    st.write(f"SA completed in {elapsed:.2f} seconds")
    st.write(f"Final Best Cost: {best_cost}")
    return best_solution, best_cost, cost_history

# ------------------ Streamlit UI ------------------
st.subheader("Parameters")
temp = st.slider("Initial Temperature", 500, 3000, 1500, step=100)
cool = st.slider("Cooling Rate", 0.85, 0.99, 0.95, step=0.01)
min_temp = st.number_input("Minimum Temperature", 1, 50, 1)
alpha = st.slider("Weight Capacity Violation (α)", 10, 100, 50)
beta = st.slider("Weight Wasted Capacity (β)", 1, 20, 5)
max_iter = st.number_input("Maximum Iterations", 10, 500, 100)

if st.button("Run Simulated Annealing"):
    best_schedule, best_cost, history = simulated_annealing(temp, cool, min_temp, alpha, beta, max_iter)
    
    # Display final schedule
    df = pd.DataFrame([{"Exam ID": e, "Students": exam_students[e], "Classroom": r, "Room Capacity": room_capacity[r]} for e,r in best_schedule.items()])
    st.dataframe(df)

    # Plot convergence
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_title("Convergence Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Cost")
    plt.tight_layout()
    plt_file = os.path.join(BASE_DIR, "sa_convergence.png")
    plt.savefig(plt_file)
    st.pyplot(fig)
    st.success(f"Convergence plot saved to {plt_file}")
