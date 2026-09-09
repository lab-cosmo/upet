.. image:: /../static/images/upet-logo-with-text.svg
   :class: only-light
   :width: 600px

.. image:: /../static/images/upet-logo-with-text-dark.svg
   :class: only-dark
   :width: 600px

**UPET** is a family of universal interatomic potentials for advanced
materials modeling across the periodic table. These models are based on the
**Point Edge Transformer (PET)** architecture, trained on a variety of
popular atomistic datasets, and capable of predicting energies and forces
in complex atomistic workflows.

The package also ships **PET-MAD-DOS**, a universal model for predicting
the electronic density of states (DOS) of materials and molecules, as well
as their Fermi levels and bandgaps. PET-MAD-DOS uses a slightly modified
PET architecture and is trained on the MAD dataset.

.. note::

   The **PET-MAD-1.6** models, trained for 102 elements at the r2SCAN level
   of theory, are now available. On top of the MAD-1.5 training data, they
   were additionally trained on catalytic surfaces, and therefore have
   better accuracy for surface reactions and adsorption energies. They also
   come in a new **M** size. See :ref:`models` and the updated 
   `preprint <https://arxiv.org/abs/2603.02089>`_ for more details.

   .. code-block:: python

      from upet.ase import UPETCalculator
      calculator = UPETCalculator(model="pet-mad-s", version="1.6.0", device="cuda")

.. note::

   A new experimental integration of UPET with the `NVIDIA ALCHEMI Toolkit
   <https://developer.nvidia.com/alchemi>`_ is now available. It allows for
   GPU-native batched inference, relaxations and MD simulations with a
   ``torch.compile``\ d version of UPET. See :ref:`usage_nvalchemi`.

   .. code-block:: python

      from upet.nvalchemi import UPETWrapper
      model = UPETWrapper.from_checkpoint(
          model="pet-mad-s", version="1.6.0", device="cuda"
      )


Key features
------------

- **Universality**: UPET models are generally applicable, and can be used
  for predicting energies and forces, as well as the density of states,
  Fermi levels, and bandgaps for a wide range of materials and molecules.
- **Accuracy**: UPET models achieve excellent accuracies in various types
  of atomistic simulations of organic and inorganic systems.
- **Efficiency**: UPET models are highly computationally efficient and
  have low memory usage, which makes them suitable for large-scale
  simulations.
- **Infrastructure**: Various MD engines are available for diverse
  research and application needs.
- **HPC compatibility**: Efficient in HPC environments for extensive
  simulations.


Maintainers
-----------

This project is `maintained
<https://github.com/lab-cosmo/.github/blob/main/Maintainers.md>`_ by
`@abmazitov <https://github.com/abmazitov>`_, who will reply to issues and
pull requests opened on the repository as soon as possible. You can mention
them directly if you have not received an answer after a couple of days.


.. toctree::
   :maxdepth: 1
   :caption: Contents
   :hidden:

   quickstart
   installation
   usage/index
   models
   fine-tuning
   api
   generated_examples/index
   miscellaneous
   faq
   changelog
   cite
