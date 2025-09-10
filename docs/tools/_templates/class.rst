{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :inherited-members:

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

.. include:: ../../gallery/backreference/{{module}}.{{objname}}.examples

.. raw:: html

    <div style='clear:both'></div>