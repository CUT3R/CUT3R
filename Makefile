# Default SHELL used by make
SHELL=/bin/bash
# Remove message: "make[1]: Entering directory"
MAKEFLAGS += --no-print-directory
# Docker
DOCKERFILE=Dockerfile
IMAGE_TAG=sculptee:cut3r
NAME=sculptee_cut3r
# SHELL used to exec in the container
EXEC_SHELL=/bin/bash
# [Docker - Run] If true, it will run the permissions settings
SUDO_LESS=false

help: ##                                         [Usage] make your-command ARG=value
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'


update: ##-------------------------------------- [Docker - Pull] Update the Perception image by pulling the latest version from ECR
	@echo --- Update the Perception image ---
	@echo Troubleshooting: If you face any AWS ECR permission/login issue please run: make docker-login
	@echo
	docker pull $(IMAGE_TAG)
build-image: ##--------------------------------- [Docker - Build] Build the Perception image based on DOCKERFILE
	@echo --- Build the Perception Image ---
	bash pkgs/docker/build_docker.sh $(DOCKERFILE) $(IMAGE_TAG)
run: ##----------------------------------------- [Docker - Run] Run the perception container
                                                 ##Usage: make run EXEC_SHELL=zsh
	make delete
	@echo --- Run the Perception container ---
	bash pkgs/docker/run_perception_container.sh $(IMAGE_TAG) $(NAME) $(EXEC_SHELL) $(SUDO_LESS)
	make exec NAME=$(NAME) EXEC_SHELL=$(EXEC_SHELL) SUDO_LESS=$(SUDO_LESS)

exec: ##---------------------------------------- [Docker - Exec] Enter inside the container
                                                 ##make exec EXEC_SHELL=zsh NAME=docker2
                                                 ##Specify the name of the container:  make exec NAME=container2
                                                 ##Run a different shell type (zsh or sh):  make exec EXEC_SHELL=zsh
                                                 ##Skip the filesystem permissions check that requires sudo:  make exec SUDO_LESS=true
	@echo --- Exec dev container ---
	# E.g.: make exec SHELL=zsh
	@if [ $(SUDO_LESS) = "true" ]; then \
		docker exec -it $(NAME) $(EXEC_SHELL); \
	else \
		docker exec -it $(NAME) bash -c "source /opt/container/container-entrypoint-base.sh $(EXEC_SHELL)"; \
	fi
stop: ##---------------------------------------- [Docker - Stop] Delete the container (you will loose any data that is not in ~/dev/summer_robotics)
	@echo --- Stop dev container ---
	docker stop $(NAME)
delete: ##-------------------------------------- [Docker - Delete] Delete the container (you will loose any data that is not in ~/dev/summer_robotics)
	@echo --- Delete old container ---
	docker ps -q --filter "name=$(NAME)" | grep -q . && docker rm -f $(NAME) || true
	@echo