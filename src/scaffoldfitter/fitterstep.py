"""
Base class for fitter steps.
"""
import abc


class FitterStep:
    """
    Base class for fitter steps.
    """
    _defaultGroupName = "<default>"

    def __init__(self):
        """
        Construct base class.
        """
        self._fitter = None  # set by subsequent client call to Fitter.addFitterStep()
        self._hasRun = False
        # Fitter steps store many settings in a structure which allows them to
        # be specified per-group. The internal data structure for this is a map
        # from group name to a dict of named settings.
        # The group name is taken from the model as data group names differing
        # by case or whitespace are set by Fitter to matching model names.
        # A special group name given by self.getDefaultGroupName() maps to the
        # default settings used where not specified per-group.
        # Values can be inherited from earlier steps of the same type, however
        # the special value None cancels previous value to restore the default.
        self._groupSettings = {}

    @classmethod
    def getDefaultGroupName(cls):
        return cls._defaultGroupName

    def getFitter(self):
        return self._fitter

    def setFitter(self, fitter):
        """
        Should only be called by Fitter when adding or removing from it.
        """
        self._fitter = fitter

    @classmethod
    @abc.abstractmethod
    def getJsonTypeId(cls):
        pass

    def decodeSettingsJSONDict(self, dctIn: dict):
        """
        Decode definition of step from JSON dict.
        """
        assert self.getJsonTypeId() in dctIn
        # update group settings first
        groupSettingsIn = dctIn.get("groupSettings")
        if groupSettingsIn:
            self._groupSettings.update(groupSettingsIn)

    def encodeSettingsJSONDict(self) -> dict:
        """
        Encode definition of step in dict.
        :return: Settings in a dict ready for passing to json.dump.
        """
        return {
            self.getJsonTypeId(): True,
            "groupSettings": self._groupSettings
            }

    def getGroupSettingsNames(self):
        """
        :return:  List of names of groups settings are held for.
        """
        return list(self._groupSettings.keys())

    def clearGroupSetting(self, groupName: str, settingName: str):
        """
        Clear setting for group, removing group settings dict if empty.
        :param groupName:  Exact model group name, or None for default group.
        :param settingName: Exact setting name.
        """
        if groupName is None:
            groupName = self._defaultGroupName
        groupSettings = self._groupSettings.get(groupName)
        if groupSettings:
            groupSettings.pop(settingName, None)
            if len(groupSettings) == 0:
                self._groupSettings.pop(groupName)

    def _getInheritedGroupSetting(self, groupName: str, settingName: str):
        """
        :param groupName:  Exact model group name, or None for default group.
        :param settingName: Exact setting name.
        :return: Inherited value or None if none.
        """
        if groupName is None:
            groupName = self._defaultGroupName
        inheritStep = self
        while True:
            inheritStep = self.getFitter().getInheritFitterStep(inheritStep)
            if not inheritStep:
                return None
            groupSettings = inheritStep._groupSettings.get(groupName)
            if groupSettings:
                inheritedValue = groupSettings.get(settingName, "<not set>")
                if inheritedValue != "<not set>":
                    return inheritedValue

    def getGroupSetting(self, groupName: str, settingName: str, defaultValue):
        """
        Get group setting of supplied name, with reset & inherit ability.
        :param groupName:  Exact model group name, or None for default group.
        :param settingName: Exact setting name.
        :param defaultValue: Value to use if setting not found for group.
        Value falls back to value for default group if not set or inherited,
        or defaultValue if not set in default group.
        The second return value is True if the value is set locally to a value
        or None if reset locally.
        The third return value is True if a previous config has set the value.
        """
        if groupName is None:
            groupName = self._defaultGroupName
        value = None
        setLocally = False
        groupSettings = self._groupSettings.get(groupName)
        if groupSettings:
            if settingName in groupSettings:
                value = groupSettings[settingName]
                setLocally = None if (value is None) else True
        inheritedValue = self._getInheritedGroupSetting(groupName, settingName)
        inheritable = inheritedValue is not None
        if inheritable and not (setLocally or (setLocally is None)):
            value = inheritedValue
        if value is None:
            if groupName != self._defaultGroupName:
                value = self.getGroupSetting(self._defaultGroupName, settingName, defaultValue)[0]
            else:
                value = defaultValue
        return value, setLocally, inheritable

    def setGroupSetting(self, groupName: str, settingName: str, value):
        """
        Set value of setting or None to reset to default.
        :param groupName:  Exact model group name, or None for default group.
        :param settingName: Exact setting name.
        :param value: Value to assign, or None to reset. If the value is not
        inherited, None will clear the setting. Caller must check valid value.
        """
        if groupName is None:
            groupName = self._defaultGroupName
        if value is None:
            if self._getInheritedGroupSetting(groupName, settingName) is None:
                self.clearGroupSetting(groupName, settingName)
                return
        groupSettings = self._groupSettings.get(groupName)
        if not groupSettings:
            groupSettings = self._groupSettings[groupName] = {}
        groupSettings[settingName] = value

    def hasRun(self):
        return self._hasRun

    def setHasRun(self, hasRun):
        self._hasRun = hasRun

    def getDiagnosticLevel(self):
        return self._fitter.getDiagnosticLevel()

    def run(self, modelFileNameStem=None):
        """
        Override to perform action of derived FitStep
        :param modelFileNameStem: Optional name stem of intermediate output file to write.
        """
        pass
