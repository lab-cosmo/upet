.. _usage_gromacs:

GROMACS
=======

.. note::

   Full documentation for the UPET integration with GROMACS is work in progress.

UPET models can be used with `GROMACS <https://www.gromacs.org/>`_ via
the ``metatomic`` interface. See the `metatomic GROMACS documentation
<https://docs.metatensor.org/metatomic/latest/engines/gromacs.html>`_ for the
general workflow; exporting a UPET checkpoint with ``mtt export`` (see
:ref:`usage_metatrain`) produces a TorchScript model that can be plugged
directly into the GROMACS driver.
