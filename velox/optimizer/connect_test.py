import socket
import json

def mcts_function(client_socket, input_json):
    # Example: Implement your Monte Carlo Tree Search function here
    # This function should receive a list of strings and send/receive data to/from the C++ client
    input_strings = input_json["strings"]
    # Example: Send each string to C++ and receive an integer
    print(f"Received start strings from C++: {input_strings}")
    # output_strings = ["Merge2Single"]
    # # only one string
    # # output_strings = ["output1"]
    # joined_output = "#".join(output_strings) + "#E"
    # client_socket.sendall(joined_output.encode('utf-8'))
    output_json = {
        "action_strings": ["Merge2Single"],
        "end": "F",
        # You can add more data to the JSON object as needed
    }
    output_str = json.dumps(output_json)
    client_socket.sendall(output_str.encode('utf-8'))

        # Receive the integer result from C++
    result = int.from_bytes(client_socket.recv(4), byteorder='big')
    print("Received integer from C++:", result)

    # MCTS produces a list of output strings

    # Signal the end of the list
    final_json = {
        "action_strings": ["Merge2Single"],
        "end": "T",
        # You can add more data to the JSON object as needed
    }
    final_str = json.dumps(final_json)
    client_socket.sendall(final_str.encode('utf-8'))

# Set up the server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 12345)

server_socket.bind(server_address)
server_socket.listen(1)

print("Python server listening on port 12345...")

while True:
    # Accept incoming connection from C++
    client_socket, client_address = server_socket.accept()
    print(f"Connected to C++ client: {client_address}")

    # Receive the start string from C++
    start_str = client_socket.recv(1024).decode('utf-8')
    print(f"Received start string from C++: {start_str}")

    if start_str == "start":
        # Receive the list of strings from C++
        # input_strings = client_socket.recv(1024).decode('utf-8').split('#')[:-1]
        json_str = client_socket.recv(1024).decode('utf-8')
        input_json = json.loads(json_str)
        mcts_function(client_socket, input_json)
        # Run the MCTS function
        # mcts_function(client_socket, input_strings)
    else:
        print("Invalid start string received. Expected 'start'.")

    # Close the connection
    client_socket.close()
    print("Connection closed.")