"""
Fit step for configuring subsequent behaviour, e.g. data projection settings.
"""

from scaffoldfitter.fitterstep import FitterStep
import sys


class FitterStepConfig(FitterStep):

    _jsonTypeId = "_FitterStepConfig"
    _centralProjectionToken = "centralProjection"
    _dataProportionToken = "dataProportion"

    def __init__(self):
        super(FitterStepConfig, self).__init__()
        self._projectionCentreGroups = False

    @classmethod
    def getJsonTypeId(cls):
        return cls._jsonTypeId

    def decodeSettingsJSONDict(self, dctIn: dict):
        """
        Decode definition of step from JSON dict.
        """
        super().decodeSettingsJSONDict(dctIn)  # to decode group settings
        # migrate legacy settings
        projectionCentreGroups = dctIn.get("projectionCentreGroups")
        if projectionCentreGroups is not None:
            self.setProjectionCentreGroups(projectionCentreGroups)

    def encodeSettingsJSONDict(self) -> dict:
        """
        Encode definition of step in dict.
        :return: Settings in a dict ready for passing to json.dump.
        """
        # only has group settings for now
        return super().encodeSettingsJSONDict()

    def clearGroupCentralProjection(self, groupName):
        """
        Clear local group central projection so fall back to last config or global default.
        :param groupName:  Exact model group name, or None for default group.
        """
        self.clearGroupSetting(groupName, self._centralProjectionToken)

    def getGroupCentralProjection(self, groupName):
        """
        Get flag controlling whether projections for this group are made with
        data points and target elements translated to a common central point.
        This can help fit groups which start well away from their targets.
        If not set or inherited, gets value from default group.
        :param groupName:  Exact model group name, or None for default group.
        :return:  Central projection flag, setLocally, inheritable.
        The second return value is True if the value is set locally to a value
        or None if reset locally.
        The third return value is True if a previous config has set the value.
        """
        return self.getGroupSetting(groupName, self._centralProjectionToken, False)

    def setGroupCentralProjection(self, groupName, centralProjection):
        """
        Set flag controlling whether projections for this group are made with
        data points and target elements translated to a common central point.
        This can help fit groups which start well away from their targets.
        :param groupName:  Exact model group name, or None for default group.
        :param centralProjection:  Boolean True/False or None to reset to global
        default. Function ensures value is valid.
        """
        if centralProjection is not None:
            if not isinstance(centralProjection, bool):
                centralProjection = False
        self.setGroupSetting(groupName, self._centralProjectionToken, centralProjection)

    def isProjectionCentreGroups(self):
        """
        :deprecated: use isGroupCentralProjection.
        """
        print("Config isProjectionCentreGroups is deprecated", file=sys.stderr)
        return self.getGroupCentralProjection(None)[0]

    def setProjectionCentreGroups(self, projectionCentreGroups):
        """
        :deprecated: use setGroupCentralProjection.
        """
        print("Config setProjectionCentreGroups is deprecated", file=sys.stderr)
        self.setGroupCentralProjection(None, projectionCentreGroups)

    def clearGroupDataProportion(self, groupName):
        """
        Clear local group data proportion so fall back to last config or global default.
        :param groupName:  Exact model group name, or None for default group.
        """
        self.clearGroupSetting(groupName, self._dataProportionToken)

    def getGroupDataProportion(self, groupName):
        """
        Get proportion of group data points to include in fit, from 0.0 (0%) to
        1.0 (100%), plus flags indicating where it has been set.
        If not set or inherited, gets value from default group.
        :param groupName:  Exact model group name, or None for default group.
        :return:  Proportion, setLocally, inheritable.
        Proportion of points used for group from 0.0 to 1.0 (default).
        The second return value is True if the value is set locally to a value
        or None if reset locally.
        The third return value is True if a previous config has set the value.
        """
        return self.getGroupSetting(groupName, self._dataProportionToken, 1.0)

    def setGroupDataProportion(self, groupName, proportion):
        """
        Set proportion of group data points to include in fit, or reset to
        global default.
        :param groupName:  Exact model group name, or None for default group.
        :param proportion:  Float valued proportion from 0.0 (0%) to 1.0 (100%),
        or None to reset to global default. Function ensures value is valid.
        """
        if proportion is not None:
            if not isinstance(proportion, float):
                proportion = self.getGroupDataProportion(groupName)[0]
            elif proportion < 0.0:
                proportion = 0.0
            elif proportion > 1.0:
                proportion = 1.0
        self.setGroupSetting(groupName, self._dataProportionToken, proportion)

    def run(self, modelFileNameStem=None):
        """
        Calculate data projections with current settings.
        :param modelFileNameStem: Optional name stem of intermediate output file to write.
        """
        self._fitter.calculateDataProjections(self)
        if modelFileNameStem:
            self._fitter.writeModel(modelFileNameStem + "_config.exf")
        self.setHasRun(True)
