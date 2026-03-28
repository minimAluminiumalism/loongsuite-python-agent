LoongSuite Instrumentation for A2A
===================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/loongsuite-instrumentation-a2a.svg
   :target: https://pypi.org/project/loongsuite-instrumentation-a2a/

This library allows tracing calls to the `A2A (Agent-to-Agent) <https://github.com/a2aproject/a2a-python>`_ Python SDK.

Installation
------------

::

    pip install loongsuite-instrumentation-a2a

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.a2a import A2AInstrumentor

    A2AInstrumentor().instrument()

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `A2A Protocol <https://github.com/a2aproject/a2a-python>`_
