ngelo Download and Setup Script
# This script downloads the Neuralangelo repository from NVIDIA Labs

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
	    echo -e "${GREEN}[INFO]${NC} $1"
    }

print_error() {
	    echo -e "${RED}[ERROR]${NC} $1"
    }

print_warning() {
	    echo -e "${YELLOW}[WARNING]${NC} $1"
    }

# Check if git is installed
if ! command -v git &> /dev/null; then
	    print_error "Git is not installed. Please install git first."
	        echo "You can install it using:"
		    echo "  Ubuntu/Debian: sudo apt-get install git"
		        echo "  Fedora: sudo dnf install git"
			    echo "  Arch: sudo pacman -S git"
			        exit 1
fi

# Set the target directory
TARGET_DIR="neuralangelo"

# Check if directory already exists
if [ -d "$TARGET_DIR" ]; then
	    print_warning "Directory '$TARGET_DIR' already exists."
	        read -p "Do you want to remove it and download fresh? (y/n): " -n 1 -r
		    echo
		        if [[ $REPLY =~ ^[Yy]$ ]]; then
				        print_status "Removing existing directory..."
					        rm -rf "$TARGET_DIR"
						    else
							            print_status "Exiting without changes."
								            exit 0
									        fi
fi

# Clone the repository
print_status "Cloning Neuralangelo repository..."
git clone https://github.com/NVlabs/neuralangelo.git "$TARGET_DIR"

# Check if clone was successful
if [ $? -eq 0 ]; then
	    print_status "Successfully cloned Neuralangelo!"
	        cd "$TARGET_DIR"
		    
		    # Display repository information
		        print_status "Repository details:"
			    echo "  Location: $(pwd)"
			        echo "  Branch: $(git branch --show-current)"
				    echo "  Latest commit: $(git log -1 --format='%h - %s')"
				        
				        # Check for requirements file
					    if [ -f "requirements.txt" ]; then
						            print_status "Found requirements.txt"
							            echo "To install Python dependencies, run:"
								            echo "  pip install -r requirements.txt"
									        fi
										    
										    if [ -f "environment.yml" ] || [ -f "environment.yaml" ]; then
											            print_status "Found conda environment file"
												            echo "To create conda environment, run:"
													            echo "  conda env create -f environment.yml"
														        fi
															    
															    # Check for setup instructions
															        if [ -f "README.md" ]; then
																	        print_status "README.md found. Check it for setup instructions."
																		    fi
																		        
																		        print_status "Download complete! Navigate to '$TARGET_DIR' to get started."
																		else
																			    print_error "Failed to clone repository. Please check your internet connection and try again."
																			        exit 1
fi

# Optional: Ask if user wants to initialize submodules
if [ -f ".gitmodules" ]; then
	    print_status "This repository contains submodules."
	        read -p "Do you want to initialize submodules? (y/n): " -n 1 -r
		    echo
		        if [[ $REPLY =~ ^[Yy]$ ]]; then
				        print_status "Initializing submodules..."
					        git submodule update --init --recursive
						        print_status "Submodules initialized."
							    fi
fi

print_status "Setup complete!"
