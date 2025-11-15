from flask import Flask, render_template, request, redirect, url_for
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, time, tracemalloc, random, heapq, math, os

# --- Flask App Setup ---
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
G = None


# ---------- RANDOM GRAPH ----------
def create_random_graph():
    G = nx.DiGraph()
    nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    G.add_nodes_from(nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if random.random() < 0.5:
                weight = random.randint(1, 15)
                G.add_edge(nodes[i], nodes[j], weight=weight)
                if random.random() < 0.4:
                    G.add_edge(nodes[j], nodes[i], weight=random.randint(1, 15))
    return G

# ---------- DIJKSTRA ----------
def dijkstra(G, start, goal):
    distances = {node: float('inf') for node in G.nodes}
    previous = {node: None for node in G.nodes}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        dist, current = heapq.heappop(pq)
        if current == goal:
            break
        for neighbor in G.neighbors(current):
            new_dist = dist + G[current][neighbor]['weight']
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = current
                heapq.heappush(pq, (new_dist, neighbor))
    path, node = [], goal
    while node:
        path.insert(0, node)
        node = previous[node]
    return path, distances[goal]

# ---------- A* ----------
def a_star(G, start, goal):
    pos = nx.spring_layout(G, seed=42)
    def heuristic(a, b):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    open_set = [(0, start)]
    g = {node: float('inf') for node in G.nodes}
    f = {node: float('inf') for node in G.nodes}
    previous = {node: None for node in G.nodes}
    g[start] = 0
    f[start] = heuristic(start, goal)
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            break
        for neighbor in G.neighbors(current):
            temp_g = g[current] + G[current][neighbor]['weight']
            if temp_g < g[neighbor]:
                previous[neighbor] = current
                g[neighbor] = temp_g
                f[neighbor] = temp_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f[neighbor], neighbor))
    path, node = [], goal
    while node:
        path.insert(0, node)
        node = previous[node]
    return path, g[goal]

# ---------- AO* ----------
def ao_star(G, start, goal):
    heuristic = {node: random.randint(1, 10) for node in G.nodes}
    path = [start]
    current = start
    total_cost = 0
    while current != goal:
        neighbors = list(G.neighbors(current))
        if not neighbors:
            break
        next_node = min(neighbors, key=lambda n: G[current][n]['weight'] + heuristic[n])
        total_cost += G[current][next_node]['weight']
        path.append(next_node)
        current = next_node
    return path, total_cost

# ---------- GRAPH PLOTTING ----------
def plot_graph(G, path=None):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(7, 5))
    nx.draw(G, pos, with_labels=True, node_color="#74b9ff", node_size=1000,
            font_weight="bold", edge_color="#2d3436", width=1.8)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="darkred", font_size=9)

    if path and len(path) > 1:
        nx.draw_networkx_edges(G, pos, edgelist=list(zip(path, path[1:])),
                               edge_color="#e17055", width=3)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ---------- ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def index():
    global G
    if G is None:
        G = create_random_graph()

    result, path_img = None, None
    graph_img = plot_graph(G)

    if request.method == "POST":
        if "refresh" in request.form:
            G = create_random_graph()
            return redirect(url_for("index"))

        start = request.form["start"].upper()
        goal = request.form["goal"].upper()
        algo = request.form["algorithm"]

        if start in G.nodes and goal in G.nodes:
            tracemalloc.start()
            t1 = time.perf_counter()
            if algo == "dijkstra":
                path, cost = dijkstra(G, start, goal)
            elif algo == "astar":
                path, cost = a_star(G, start, goal)
            else:
                path, cost = ao_star(G, start, goal)
            t2 = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            elapsed = (t2 - t1) * 1000
            mem = peak / 1024
            if len(path) > 1:
                result = {
                    "algorithm": algo.upper(),
                    "path": " → ".join(path),
                    "cost": cost,
                    "time": f"{elapsed:.2f} ms",
                    "memory": f"{mem:.2f} KB"
                }
                path_img = plot_graph(G, path)
            else:
                result = {"error": "⚠️ No valid path found!"}
        else:
            result = {"error": "⚠️ Invalid nodes (A–H only)."}  # corrected range
        print("Templates directory:", os.listdir(os.path.join(os.getcwd(), 'templates')))

    return render_template("index.html", graph_img=graph_img, path_img=path_img, result=result)
# ---------- MAIN ENTRY ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns dynamic port
    app.run(host="0.0.0.0", port=port, debug=True)