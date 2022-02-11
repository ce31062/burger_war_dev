#!/usr/bin/env python
# -*- coding: utf-8 -*-

from agents.agent_conn import AgentServer

PORT = 5010

agent_server = AgentServer(PORT)
agent_server.run()

