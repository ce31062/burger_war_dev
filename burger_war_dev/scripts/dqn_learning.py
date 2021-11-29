#!/usr/bin/env python
# -*- coding: utf-8 -*-

from agents.agent_conn import AgentServer

print("dqn_learning.py")
PORT = 5010

agent_server = AgentServer(PORT)
agent_server.run()

