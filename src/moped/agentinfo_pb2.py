# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: agentinfo.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0f\x61gentinfo.proto\x12\tagentinfo\"E\n\tAgentInfo\x12\t\n\x01x\x18\x01 \x03(\x01\x12\t\n\x01y\x18\x02 \x03(\x01\x12\x0f\n\x07\x61gentId\x18\x03 \x01(\x11\x12\x11\n\tagentType\x18\x04 \x01(\x11\"0\n\x0fProbabilityInfo\x12\x0c\n\x04prob\x18\x01 \x01(\x01\x12\x0f\n\x07\x61gentId\x18\x02 \x01(\x11\"<\n\x11PredictionRequest\x12\'\n\tagentInfo\x18\x01 \x03(\x0b\x32\x14.agentinfo.AgentInfo\"k\n\x12PredictionResponse\x12\'\n\tagentInfo\x18\x01 \x03(\x0b\x32\x14.agentinfo.AgentInfo\x12,\n\x08probInfo\x18\x02 \x03(\x0b\x32\x1a.agentinfo.ProbabilityInfo2\\\n\x10MotionPrediction\x12H\n\x07Predict\x12\x1c.agentinfo.PredictionRequest\x1a\x1d.agentinfo.PredictionResponse\"\x00\x62\x06proto3')



_AGENTINFO = DESCRIPTOR.message_types_by_name['AgentInfo']
_PROBABILITYINFO = DESCRIPTOR.message_types_by_name['ProbabilityInfo']
_PREDICTIONREQUEST = DESCRIPTOR.message_types_by_name['PredictionRequest']
_PREDICTIONRESPONSE = DESCRIPTOR.message_types_by_name['PredictionResponse']
AgentInfo = _reflection.GeneratedProtocolMessageType('AgentInfo', (_message.Message,), {
  'DESCRIPTOR' : _AGENTINFO,
  '__module__' : 'agentinfo_pb2'
  # @@protoc_insertion_point(class_scope:agentinfo.AgentInfo)
  })
_sym_db.RegisterMessage(AgentInfo)

ProbabilityInfo = _reflection.GeneratedProtocolMessageType('ProbabilityInfo', (_message.Message,), {
  'DESCRIPTOR' : _PROBABILITYINFO,
  '__module__' : 'agentinfo_pb2'
  # @@protoc_insertion_point(class_scope:agentinfo.ProbabilityInfo)
  })
_sym_db.RegisterMessage(ProbabilityInfo)

PredictionRequest = _reflection.GeneratedProtocolMessageType('PredictionRequest', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTIONREQUEST,
  '__module__' : 'agentinfo_pb2'
  # @@protoc_insertion_point(class_scope:agentinfo.PredictionRequest)
  })
_sym_db.RegisterMessage(PredictionRequest)

PredictionResponse = _reflection.GeneratedProtocolMessageType('PredictionResponse', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTIONRESPONSE,
  '__module__' : 'agentinfo_pb2'
  # @@protoc_insertion_point(class_scope:agentinfo.PredictionResponse)
  })
_sym_db.RegisterMessage(PredictionResponse)

_MOTIONPREDICTION = DESCRIPTOR.services_by_name['MotionPrediction']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _AGENTINFO._serialized_start=30
  _AGENTINFO._serialized_end=99
  _PROBABILITYINFO._serialized_start=101
  _PROBABILITYINFO._serialized_end=149
  _PREDICTIONREQUEST._serialized_start=151
  _PREDICTIONREQUEST._serialized_end=211
  _PREDICTIONRESPONSE._serialized_start=213
  _PREDICTIONRESPONSE._serialized_end=320
  _MOTIONPREDICTION._serialized_start=322
  _MOTIONPREDICTION._serialized_end=414
# @@protoc_insertion_point(module_scope)
