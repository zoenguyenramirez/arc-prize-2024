#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Function to display the menu
show_menu() {
    echo "Please select an SSH environment:"
    echo "1) ssh1"
    echo "2) ssh2"
    echo "3) ssh3"
    echo "q) Quit"
}

# Function to get user input
get_choice() {
    local choice
    read -p "Enter choice [1-3 or q]: " choice
    case $choice in
        1) echo "ssh1" ;;
        2) echo "ssh2" ;;
        3) echo "ssh3" ;;
        q) echo "quit" ;;
        *) echo "invalid" ;;
    esac
}

# Main script logic
while true; do
    show_menu
    choice=$(get_choice)

    case $choice in
        ssh1|ssh2|ssh3)
            export ENV_SELECTOR=$choice
            echo "ENV_SELECTOR set to" && figlet -f standard "$(for i in {1..10}; do echo -n "${ENV_SELECTOR: -1} "; done)"

            # Get the env file path using select_env.sh
            SELECTED_ENV_FILE=$(bash "$SCRIPT_DIR/select_env.sh")
            
            # Check if select_env.sh was successful
            if [ $? -eq 0 ]; then
                echo -e "\nSelected environment file contents:"
                echo "================================"
                cat "$SELECTED_ENV_FILE"
                echo "================================"
            else
                echo "Error: Failed to get environment file path"
            fi

            break
            ;;
        quit)
            echo "Exiting without setting ENV_SELECTOR"
            exit 0
            ;;
        invalid)
            echo "Invalid option, please try again"
            ;;
    esac
done

# Optionally, you can source this script to set ENV_SELECTOR in the current shell
# Add the following line at the end of the script:
# echo "export ENV_SELECTOR=$ENV_SELECTOR"